from typing import Any, Dict, Optional

import torch
from torch import nn
import logging
from functools import partial
from tqdm import trange

import math
from dawn.models.modules.gc_sampling import *
from dawn.models.modules.diffusion.diffusion_transformer import DiffusionTransformer
from dawn.models.action_expert.utils import *

from typing import Optional, Tuple
from transformers import AutoModel, AutoProcessor, AutoImageProcessor
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

@torch.no_grad()
def sample_ddim(
    model, 
    state, 
    action, 
    goal, 
    sigmas, 
    scaler=None,
    extra_args=None, 
    callback=None, 
    disable=None, 
    eta=1., 
):
    """
    DPM-Solver 1( or DDIM sampler"""
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        # predict the next action
        denoised = model.step(state, action, goal, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'action': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
        # print(f"Step {i}, min: {action.min()}, max: {action.max()}, mean: {action.mean()}")
    return action



class TransformerDiffusionPolicy(nn.Module):
    def __init__(
        self, 
        in_channels=5,
        action_dim=7, 
        obs_dim=768, 
        goal_dim=768, 
        num_latents=224, 
        goal_window_size = 1, 
        obs_seq_len=1, 
        act_seq_len=10, 
        proprio_dim=8,
        noise_scheduler: str = 'exponential',
        sigma_sample_density_type: str = 'loglogistic',
        sampler_type: str = 'ddim',
        num_sampling_steps: int = 10,
        sigma_data: float = 0.5,
        sigma_min: float = 0.001,
        sigma_max: float = 80,
        
            
    ):
        super().__init__()

        logger.info(f"Initializing {__class__.__name__}.")
        
        self.inner_model = DiffusionTransformer(
            action_dim = action_dim,
            obs_dim = obs_dim,
            goal_dim = goal_dim,
            proprio_dim= proprio_dim,
            goal_conditioned = True,
            embed_dim = 768,
            n_dec_layers = 4,
            n_enc_layers = 4,
            n_obs_token = num_latents,
            goal_seq_len = goal_window_size,
            obs_seq_len = obs_seq_len,
            action_seq_len =act_seq_len,
            embed_pdrob = 0,
            goal_drop = 0,
            attn_pdrop = 0.3,
            resid_pdrop = 0.1,
            mlp_pdrop = 0.05,
            n_heads= 8,
            use_mlp_goal = True,
        )


        self.sigma_data = sigma_data
        self.sigma_sample_density_type = sigma_sample_density_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_sampling_steps = num_sampling_steps
        self.sampler_type = sampler_type
        self.noise_scheduler = noise_scheduler
        self.act_window_size = act_seq_len
        self.action_dim = action_dim
        self.criterion = torch.nn.functional.mse_loss  # Assuming MSE loss for action classification

        self.generator = torch.Generator(device=self.device).manual_seed(0)

        # self.visual_proj = nn.Linear(obs_dim, 768)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, 
                batch_data: Dict[str, Any],
                encoder_outputs: Optional[Dict[str, Any]] = None,
                **kwargs: Any):


        feats = []
        if "view_image_feat" in encoder_outputs:
            for view_name, feat in encoder_outputs["view_image_feat"].items():
                feats.append(feat)
        # if "text_feat" in encoder_outputs:
        #     feats.append(encoder_outputs["text_feat"])
        if "pixel_motion_feat" in encoder_outputs:
            feats.append(encoder_outputs["pixel_motion_feat"])
        
        visual_feat = torch.cat(feats, dim=1)  # [B, N, C]
        # visual_feat = self.visual_proj(visual_feat)
    
        perceptual_emb = {
            'state_images': visual_feat,  # [B, N, C]
            'modality': "language"
        }

        # if "robot_pose" in x:
        #     perceptual_emb["state_obs"] = x["robot_pose"] #B,1,8

        latent_goal = encoder_outputs["text_feat"]
        actions = batch_data.get("action", None)

        if not self.training:
            return self.eval_forward(perceptual_emb, latent_goal, actions)
        
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.loss(perceptual_emb, actions, latent_goal, noise, sigmas)

        outputs = {
            "total_loss": loss,

        }
        return outputs
    
    def loss(self, state, action, goal, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output = self.inner_model(state, noised_input * c_in, goal, sigma, **kwargs)
        target = (action - c_skip * noised_input) / c_out
        
        loss = (model_output - target).pow(2).flatten(1).mean()
        return loss, model_output 

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)  # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')
    
    def sample_loop(
            self,
            sigmas,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            latent_plan: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        # ODE deterministic
        elif sampler_type == 'euler':
            x_0 = sample_euler(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.inner_model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0
    
    def step(self, state, action, goal, sigma, **kwargs):
        """
        Perform the forward pass of the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(state, action * c_in, goal, sigma, **kwargs) * c_out + action * c_skip
    
    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def eval_forward(
        self, 
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        labels: torch.Tensor=None,

    ):
        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            perceptual_emb,
            latent_goal,
            inference=True,
        )
        return_dict = { "logits": act_seq}

        if labels is not None:
            # If labels are provided, compute the loss

            loss = self.criterion(act_seq.flatten(start_dim=1), labels.flatten(start_dim=1))
            return_dict["total_loss"] = loss

            trans_loss = F.mse_loss(act_seq[:, :3].flatten(start_dim=1), labels[:, :3].flatten(start_dim=1))
            rot_loss = F.mse_loss(act_seq[:, 3:6].flatten(start_dim=1), labels[:, 3:6].flatten(start_dim=1))
            gripper_loss = F.mse_loss(act_seq[:, 6].flatten(start_dim=1), labels[:, 6].flatten(start_dim=1))
            l1_gripper_loss = F.l1_loss(act_seq[:, 6].flatten(start_dim=1), labels[:, 6].flatten(start_dim=1))
            
            return_dict.update({
                "trans_loss": trans_loss,
                "rot_loss": rot_loss,
                "gripper_loss": gripper_loss,
                "l1_gripper_loss": l1_gripper_loss,
            })

        return return_dict
        
    def denoise_actions(  # type: ignore
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            inference: Optional[bool] = False,
            extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.inner_model.eval()
        if len(latent_goal.shape) < len(
                perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
            latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)

        # self.generator = torch.Generator(device=self.device).manual_seed(0)
        x = torch.randn((len(latent_goal), self.act_window_size, self.action_dim), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions

if __name__ == "__main__":
    model = TransformerDiffusionPolicy(device="cuda")