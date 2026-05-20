from typing import Dict, Optional, Tuple, Union
from diffusers.models import UNetSpatioTemporalConditionModel
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import random
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import numpy as np

class Diffusion_feature_extractor(nn.Module):
    def __init__(
        self,
        pipeline=None,
    ):
        super().__init__()
        self.pipeline = pipeline if pipeline is not None else StableVideoDiffusionPipeline()
        self.num_frames = 4

    @torch.no_grad()
    def forward(
            self,
            pixel_values: torch.Tensor,
            encoder_hidden_states,
            timestep: Union[torch.Tensor, float, int],
            extract_layer_idx: Union[torch.Tensor, float, int],
            use_latent = False,
            all_layer = False,
            step_time = 1,
            max_length = 20,
    ):

        height = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor //3
        width = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor //3
        self.pipeline.vae.eval()
        self.pipeline.image_encoder.eval()
        device = self.pipeline.unet.device
        dtype = self.pipeline.vae.dtype
        #print('dtype:',dtype)
        vae = self.pipeline.vae

        num_videos_per_prompt=1

        batch_size = pixel_values.shape[0]

        pixel_values = rearrange(pixel_values, 'b f c h w-> (b f) c h w').to(dtype)

        # with torch.no_grad():
        #     # texts, tokenizer, text_encoder, img_cond=None, img_cond_mask=None, img_encoder=None, position_encode=True, use_clip=False, max_length=20
        #     encoder_hidden_states = self.encode_text(texts, self.tokenizer, self.text_encoder, position_encode=self.position_encoding, use_clip=True, max_length=max_length)
        # encoder_hidden_states = encoder_hidden_states.to(dtype)
        image_embeddings = encoder_hidden_states

        needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast
        #if needs_upcasting:
        #    self.pipeline.vae.to(dtype=torch.float32)
        #    pixel_values.to(dtype=torch.float32)
        if pixel_values.shape[-3] == 4:
            image_latents = pixel_values/vae.config.scaling_factor
        else:
            image_latents = self.pipeline._encode_vae_image(pixel_values, device, num_videos_per_prompt, False)
        image_latents = image_latents.to(image_embeddings.dtype)

        print('size:', image_latents.shape)

        #if needs_upcasting:
        #    self.pipeline.vae.to(dtype=torch.float16)

        #num_frames = self.pipeline.unet.config.num_frames
        num_frames = 4
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        fps=4
        motion_bucket_id=127
        added_time_ids = self.pipeline._get_add_time_ids(
            fps,
            motion_bucket_id,
            0,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            False,
        )
        added_time_ids = added_time_ids.to(device)

        self.pipeline.scheduler.set_timesteps(timestep, device=device)
        timesteps = self.pipeline.scheduler.timesteps

        num_channels_latents = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            None,
            None,
        )

        # print(latents.shape, image_latents.shape, image_embeddings.shape)
        for i, t in enumerate(timesteps):
            #print('step:',i)
            if i == step_time - 1:
                complete = False
            else:
                complete = True
            #print('complete:',complete)

            latent_model_input = latents
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimention
            # latent_model_input = torch.cat([mask, latent_model_input, image_latents], dim=2)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            #print('latent_model_input_shape:',latent_model_input.shape)
            #print('image_embeddings_shape:',image_embeddings.shape)

            # predict the noise residual
            # print('extract_layer_idx:',extract_layer_idx)
            # print('latent_model_input_shape:',latent_model_input.shape)
            # print('encoder_hidden_states:',image_embeddings.shape)
            feature_pred = self.step_unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                use_layer_idx=extract_layer_idx,
                all_layer = all_layer,
                complete = complete,
            )[0]
            # feature_pred = self.pipeline.unet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=image_embeddings,
            #     added_time_ids=added_time_ids,
            #     return_dict=False,
            # )[0]

            # print('feature_pred_shape:',feature_pred.shape)

            if not complete:
                break

            latents = self.pipeline.scheduler.step(feature_pred, t, latents).prev_sample

        return feature_pred

    def step_unet(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        use_layer_idx: int = 5,
        all_layer: bool = False,
        complete: bool = False,
    ) :
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.pipeline.unet.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.pipeline.unet.time_embedding(t_emb)

        time_embeds = self.pipeline.unet.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.pipeline.unet.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.pipeline.unet.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.pipeline.unet.down_blocks:
            #print('sample_shape:',sample.shape)
            #print('emb_shape:', emb.shape)
            #print('encoder_hidden_states_shape:', encoder_hidden_states.shape)
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.pipeline.unet.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        feature_list = []

        # 5. up
        for i, upsample_block in enumerate(self.pipeline.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )
            if i < use_layer_idx:
                factor = 2**(use_layer_idx - i)
                feature_list.append(torch.nn.functional.interpolate(sample,scale_factor=factor))
            #print('up_sample_idx:',i)
            if i == use_layer_idx and not complete:
                feature_list.append(sample)
                break

        if not complete:
            if all_layer:
                sample = torch.cat(feature_list, dim=1)
                sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
            else:
                sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
            # 6. post-process
            return (sample,)

        else:
            sample = self.pipeline.unet.conv_norm_out(sample)
            sample = self.pipeline.unet.conv_act(sample)
            sample = self.pipeline.unet.conv_out(sample)

            # 7. Reshape back to original shape
            sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

            return (sample,)
