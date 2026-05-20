
import kornia
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from torchvision.utils import flow_to_image


def flow_cart_to_polar(flow, mag_norm=64.0):
    if flow.ndim == 3:
        fx, fy = flow[0], flow[1]
        channel_dim = 0
    elif flow.ndim == 4:
        fx, fy = flow[:, 0], flow[:, 1]
        channel_dim = 1
    else:
        raise ValueError(f"Expected flow shape (2,H,W) or (B,2,H,W), got {tuple(flow.shape)}.")

    magnitude = torch.sqrt(fx**2 + fy**2) / mag_norm  # Normalize magnitude to [0, 1]
    magnitude = torch.clamp(magnitude, min=0.0, max=1.0)
    angle = torch.arctan2(fy, fx)  # range: [-pi, pi]
    angle = angle + torch.pi  # Shift angle to [0, 2pi]
    return torch.stack([magnitude, angle], axis=channel_dim)


def flow_polar_to_cart(magnitude, angle, mag_norm=64.0):
    angle_rad = angle - torch.pi
    fx = magnitude * torch.cos(angle_rad) * mag_norm
    fy = magnitude * torch.sin(angle_rad) * mag_norm
    if len(magnitude.shape) == 2:  # no batch dimension
        return torch.stack([fx, fy], axis=0)
    else:  # len == 3
        return torch.stack([fx, fy], axis=1)


def flow_cart_to_hsv(flow, mag_norm=64.0):
    polar = flow_cart_to_polar(flow, mag_norm=mag_norm)
    if flow.ndim == 3:
        hsv = torch.zeros((3, flow.shape[1], flow.shape[2]), dtype=flow.dtype, device=flow.device)
        hsv[0] = polar[1].to(flow.dtype)  # Angle as Hue
        hsv[1] = polar[0].to(flow.dtype)  # Mag as Saturation
        hsv[2] = (polar[0] + (1 - polar[0])).to(flow.dtype)  # Third channel (all ones)
    elif flow.ndim == 4:
        hsv = torch.zeros((flow.shape[0], 3, flow.shape[2], flow.shape[3]), dtype=flow.dtype, device=flow.device)
        hsv[:, 0] = polar[:, 1].to(flow.dtype)  # Angle as Hue
        hsv[:, 1] = polar[:, 0].to(flow.dtype)  # Mag as Saturation
        hsv[:, 2] = (polar[:, 0] + (1 - polar[:, 0])).to(flow.dtype)  # Third channel (all ones)
    else:
        raise ValueError(f"Expected flow shape (2,H,W) or (B,2,H,W), got {tuple(flow.shape)}.")

    rgb = kornia.color.hsv.hsv_to_rgb(hsv)
    return rgb


def flow_hsv_to_cart(rgb_array, mag_norm=64.0):
    hsv_array = kornia.color.hsv.rgb_to_hsv(rgb_array)
    if len(hsv_array.shape) == 3:  # no batch dimension
        angle = hsv_array[0]
        mag = hsv_array[1]
    else:  # len == 4
        angle = hsv_array[:, 0]
        mag = hsv_array[:, 1]
    flow_cart = flow_polar_to_cart(mag, angle, mag_norm=mag_norm)
    return flow_cart


class FlowToRGBConverter:
    SUPPORTED_MODES = ("torchvision", "mean3", "hsv")

    def __init__(self, mode="torchvision", mag_norm=64.0):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes: {self.SUPPORTED_MODES}.")
        self.mode = mode
        self.mag_norm = float(mag_norm)

    def flow_to_unit(self, flow: torch.Tensor) -> torch.Tensor:
        # Maps raw flow from [-mag_norm, mag_norm] to [0, 1].
        return (((flow / self.mag_norm) + 1.0) / 2.0).clamp(0.0, 1.0)

    def unit_to_flow(self, flow_unit: torch.Tensor) -> torch.Tensor:
        # Inverse of flow_to_unit.
        return (flow_unit * 2.0 - 1.0) * self.mag_norm

    def _torchvision(self, flow: torch.Tensor) -> torch.Tensor:
        return flow_to_image(flow).to(flow.dtype) / 255.0

    def _mean3(self, flow: torch.Tensor) -> torch.Tensor:
        flow_2ch = self.flow_to_unit(flow)
        third = flow_2ch.mean(dim=1, keepdim=True)
        return torch.cat([flow_2ch, third], dim=1)

    def _hsv(self, flow: torch.Tensor) -> torch.Tensor:
        return flow_cart_to_hsv(flow, mag_norm=self.mag_norm).clamp(0.0, 1.0)

    def _mean3_to_flow(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim not in (3, 4):
            raise ValueError(f"Expected rgb shape (3,H,W) or (B,3,H,W), got {tuple(rgb.shape)}.")
        if rgb.shape[-3] != 3:
            raise ValueError(f"Expected 3 channels for rgb input, got shape {tuple(rgb.shape)}.")
        flow_2ch = rgb[..., :2, :, :]
        return self.unit_to_flow(flow_2ch)

    def _hsv_to_flow(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim not in (3, 4):
            raise ValueError(f"Expected rgb shape (3,H,W) or (B,3,H,W), got {tuple(rgb.shape)}.")
        if rgb.shape[-3] != 3:
            raise ValueError(f"Expected 3 channels for rgb input, got shape {tuple(rgb.shape)}.")
        return flow_hsv_to_cart(rgb, mag_norm=self.mag_norm)

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        if self.mode == "torchvision":
            return self._torchvision(flow)
        if self.mode == "mean3":
            return self._mean3(flow)
        return self._hsv(flow)

    def rgb_to_flow(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.mode == "torchvision":
            raise NotImplementedError(
                "Decoding flow from torchvision flow_to_image output is not supported (many-to-one mapping)."
            )
        if self.mode == "mean3":
            return self._mean3_to_flow(rgb)
        return self._hsv_to_flow(rgb)


class PixelMotionVisualizer:
    SUPPORTED_RENDER_MODES = ("arrows", "rgb", "both")
    SUPPORTED_LAYOUT_MODES = ("overlay", "rgb_only", "side_by_side")

    def __init__(self, mag_norm=64.0, flow_rgb_mode="torchvision"):
        self.converter = FlowToRGBConverter(mode=flow_rgb_mode, mag_norm=mag_norm)

    @staticmethod
    def _to_numpy_image(image):
        if torch.is_tensor(image):
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            image = image.detach().cpu().float().numpy()
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    @staticmethod
    def _to_numpy_flow(flow):
        if torch.is_tensor(flow):
            if flow.ndim == 3 and flow.shape[0] == 2:
                flow = flow.permute(1, 2, 0)
            flow = flow.detach().cpu().float().numpy()
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError(f"Expected flow shape (H,W,2) or (2,H,W), got {tuple(flow.shape)}.")
        return flow

    def _to_numpy_flow_from_rgb(self, flow_rgb):
        if torch.is_tensor(flow_rgb):
            rgb = flow_rgb.detach().float()
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = rgb.unsqueeze(0)
            elif rgb.ndim == 4 and rgb.shape[1] == 3:
                pass
            else:
                raise ValueError(
                    f"Expected RGB flow shape (3,H,W) or (B,3,H,W), got {tuple(flow_rgb.shape)}."
                )
        else:
            rgb = torch.from_numpy(np.asarray(flow_rgb)).float()
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1).unsqueeze(0)
            elif rgb.ndim == 4 and rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Expected RGB flow shape (H,W,3) or (B,H,W,3), got {tuple(np.asarray(flow_rgb).shape)}."
                )
        flow = self.converter.rgb_to_flow(rgb)
        return self._to_numpy_flow(flow[0])

    def _flow_overlay_rgb(self, flow_np: np.ndarray) -> np.ndarray:
        flow_t = torch.from_numpy(flow_np).permute(2, 0, 1).float().unsqueeze(0)
        flow_rgb = self.converter(flow_t)[0].permute(1, 2, 0).cpu().numpy()
        return flow_rgb.clip(0.0, 1.0)

    @staticmethod
    def _draw_arrows(ax, flow_np, step, min_magnitude, quiver_color):
        h, w = flow_np.shape[:2]
        y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].astype(np.int32)
        sampled = flow_np[y, x]
        fx, fy = sampled[..., 0], sampled[..., 1]
        magnitude = np.sqrt(fx**2 + fy**2)
        mask = magnitude > float(min_magnitude)
        if np.any(mask):
            ax.quiver(
                x[mask],
                y[mask],
                fx[mask],
                fy[mask],
                color=quiver_color,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )

    @staticmethod
    def _validate_modes(render_mode, layout_mode):
        if render_mode not in PixelMotionVisualizer.SUPPORTED_RENDER_MODES:
            raise ValueError(
                f"Unsupported render_mode: {render_mode}. "
                f"Supported: {PixelMotionVisualizer.SUPPORTED_RENDER_MODES}."
            )
        if layout_mode not in PixelMotionVisualizer.SUPPORTED_LAYOUT_MODES:
            raise ValueError(
                f"Unsupported layout_mode: {layout_mode}. "
                f"Supported: {PixelMotionVisualizer.SUPPORTED_LAYOUT_MODES}."
            )

    def visualize_flow_vectors_as_pil(
        self,
        image,
        flow=None,
        flow_is_rgb=False,
        step=16,
        title="Optical Flow Vectors",
        overlay_alpha=0.6,
        min_magnitude=10.0,
        quiver_color="red",
        render_mode="both",
        layout_mode="overlay",
    ):
        self._validate_modes(render_mode, layout_mode)
        image_np = self._to_numpy_image(image)
        if flow is None:
            flow_np = None
        elif flow_is_rgb:
            flow_np = self._to_numpy_flow_from_rgb(flow)
        else:
            flow_np = self._to_numpy_flow(flow)
        flow_rgb = self._flow_overlay_rgb(flow_np) if flow_np is not None else None
        # print(f"RGB - Min: {flow_rgb.min()}, Max: {flow_rgb.max()}")
        # print(f"Flow - Min: {flow_np.min()}, Max: {flow_np.max()}")
        flow_rgb[:,:,:] = 0.5
        if layout_mode == "rgb_only" and flow_rgb is None:
            raise ValueError("`flow` is required when layout_mode='rgb_only'.")
        if layout_mode == "side_by_side" and flow_rgb is None:
            raise ValueError("`flow` is required when layout_mode='side_by_side'.")

        fig_size = (8, 4) if layout_mode == "side_by_side" else (4, 4)
        fig = Figure(figsize=fig_size, dpi=150)
        canvas = FigureCanvas(fig)

        if layout_mode == "overlay":
            ax = fig.add_subplot(111)
            ax.imshow(image_np)
            if flow_np is not None:
                if render_mode in ("rgb", "both"):
                    ax.imshow(flow_rgb, alpha=overlay_alpha)
                if render_mode in ("arrows", "both"):
                    self._draw_arrows(ax, flow_np, step, min_magnitude, quiver_color)
            ax.set_title(title)
            ax.axis("off")

        elif layout_mode == "rgb_only":
            ax = fig.add_subplot(111)
            ax.imshow(flow_rgb)
            if render_mode in ("arrows", "both"):
                self._draw_arrows(ax, flow_np, step, min_magnitude, quiver_color)
            ax.set_title(title)
            ax.axis("off")

        else:  # side_by_side
            ax_left = fig.add_subplot(1, 2, 1)
            ax_right = fig.add_subplot(1, 2, 2)

            ax_left.imshow(image_np)
            if render_mode in ("arrows", "both"):
                self._draw_arrows(ax_left, flow_np, step, min_magnitude, quiver_color)
            ax_left.set_title(f"{title} (Image)")
            ax_left.axis("off")

            ax_right.imshow(flow_rgb)
            if render_mode in ("arrows", "both"):
                self._draw_arrows(ax_right, flow_np, step, min_magnitude, quiver_color)
            ax_right.set_title(f"{title} (Flow)")
            ax_right.axis("off")

        fig.tight_layout()

        canvas.draw()
        buf = canvas.buffer_rgba()
        pil_image = Image.frombuffer("RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)
        return pil_image

    def visualize_pred_target_grid_as_pil(
        self,
        pred_image,
        pred_flow,
        pred_flow_is_rgb=False,
        target_image=None,
        target_flow=None,
        target_flow_is_rgb=False,
        step=16,
        title="Predicted vs Target Pixel Motion",
        overlay_alpha=0.6,
        min_magnitude=1.0,
        quiver_color="red",
    ):
        pred_image_np = self._to_numpy_image(pred_image)
        if pred_flow_is_rgb:
            pred_flow_np = self._to_numpy_flow_from_rgb(pred_flow)
            pred_flow_rgb = pred_flow.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        else:
            pred_flow_np = self._to_numpy_flow(pred_flow)
            pred_flow_rgb = self._flow_overlay_rgb(pred_flow_np)

        has_target = target_image is not None and target_flow is not None
        if has_target:
            target_image_np = self._to_numpy_image(target_image)
            if target_flow_is_rgb:
                target_flow_np = self._to_numpy_flow_from_rgb(target_flow)
                target_flow_rgb = target_flow.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            else:
                target_flow_np = self._to_numpy_flow(target_flow)
                target_flow_rgb = self._flow_overlay_rgb(target_flow_np)

        nrows = 2 if has_target else 1
        fig = Figure(figsize=(12, 4 * nrows), dpi=150)
        canvas = FigureCanvas(fig)

        # Row 1: prediction
        ax11 = fig.add_subplot(nrows, 3, 1)
        ax12 = fig.add_subplot(nrows, 3, 2)
        ax13 = fig.add_subplot(nrows, 3, 3)

        ax11.imshow(pred_image_np)
        ax11.set_title("Input Image")
        ax11.axis("off")

        ax12.imshow(pred_flow_rgb)
        ax12.set_title("Predicted Pixel Motion")
        if min_magnitude is not None:
            self._draw_arrows(ax12, pred_flow_np, step, min_magnitude, quiver_color)
        ax12.axis("off")

        ax13.imshow(pred_image_np)
        ax13.imshow(pred_flow_rgb, alpha=overlay_alpha)
        if min_magnitude is not None:
            self._draw_arrows(ax13, pred_flow_np, step, min_magnitude, quiver_color)
        ax13.set_title("Predicted Overlay")
        ax13.axis("off")

        # Row 2: target (optional)
        if has_target:
            ax21 = fig.add_subplot(nrows, 3, 4)
            ax22 = fig.add_subplot(nrows, 3, 5)
            ax23 = fig.add_subplot(nrows, 3, 6)

            ax21.imshow(target_image_np)
            ax21.set_title("Target Image")
            ax21.axis("off")

            ax22.imshow(target_flow_rgb)
            ax22.set_title("Target Pixel Motion")
            if min_magnitude is not None:
                self._draw_arrows(ax22, target_flow_np, step, min_magnitude, quiver_color)
            ax22.axis("off")

            ax23.imshow(target_image_np)
            ax23.imshow(target_flow_rgb, alpha=overlay_alpha)
            if min_magnitude is not None:
                self._draw_arrows(ax23, target_flow_np, step, min_magnitude, quiver_color)
            ax23.set_title("Target Overlay")
            ax23.axis("off")

        fig.suptitle(title)
        fig.tight_layout()

        canvas.draw()
        buf = canvas.buffer_rgba()
        return Image.frombuffer("RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)
