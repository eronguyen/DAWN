from torch import Tensor, nn
import torch
from torchvision.models import optical_flow
import einops

from .utils import FlowToRGBConverter


class PixelMotionEstimator(nn.Module):
    def __init__(self, flow_to_rgb=True, flow_rgb_mode="torchvision", mag_norm=64.0):
        super().__init__()
        self.flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()
        if isinstance(flow_to_rgb, str):
            # Backward compatibility: allow passing mode via legacy argument.
            flow_rgb_mode = flow_to_rgb
            flow_to_rgb = True

        self.flow_to_rgb = bool(flow_to_rgb)
        self.flow_converter = FlowToRGBConverter(mode=flow_rgb_mode, mag_norm=mag_norm)

        self.flow_model.requires_grad_(False)

    @torch.inference_mode()
    def estimate_flow(self, images: Tensor) -> Tensor:
        if images.shape[1] < 2:
            images = torch.cat([images, images], dim=1)  # Duplicate the first frame if only one frame is provided.
        flow_input, _ = self.flow_transform(images, images)
        start_im = einops.rearrange(flow_input[:, :-1], "b t c h w -> (b t) c h w")
        end_im = einops.rearrange(flow_input[:, 1:], "b t c h w -> (b t) c h w")
        with torch.no_grad():
            flow_tensor = self.flow_model(start_im, end_im, num_flow_updates=6)[-1]
                
        if self.flow_to_rgb:
            flow_tensor = self.flow_converter(flow_tensor)
        else:
            flow_tensor = self.flow_converter.flow_to_unit(flow_tensor)

        flow_tensor = einops.rearrange(flow_tensor, "(b t) c h w -> b t c h w", b=flow_input.shape[0])
        # normalize the flow
        return flow_tensor 
