from typing import Any, Dict, Optional

import torch
from torch import nn
import transformers
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
import logging

from .base import BaseActionExpert

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TransformerAction(BaseActionExpert):
    def __init__(
        self, 
        image_size=128, 
        in_channels=5, 
        action_space=7, 
        num_actions=10, 
        hidden_dim=768, 
        num_layers=6, 
        num_heads=8,
        mlp_dim=2048,
    ):
        super().__init__()
        logger.info(f"Initializing {__class__.__name__}.")
        
        # Initialize the action decoder
        self.query_embed = nn.Embedding(num_actions, hidden_dim)
        # self.query_pos = nn.Embedding(num_actions, hidden_dim)
        self.action_decoder = MLP(hidden_dim, mlp_dim, action_space, 3)

        trans_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.model = nn.TransformerDecoder(trans_layer, 6)

        self.num_actions = num_actions
        self.criterion = torch.nn.functional.mse_loss  # Assuming MSE loss for action classification
    
    def forward(self, 
                batch_data: Dict[str, Any],
                encoder_outputs: Optional[Dict[str, Any]] = None,
                **kwargs: Any):


        feats = []
        if "view_image_feat" in encoder_outputs:
            for view_name, feat in encoder_outputs["view_image_feat"].items():
                feats.append(feat)
        
        if "pixel_motion_feat" in encoder_outputs:
            feats.append(encoder_outputs["pixel_motion_feat"])
        
        if "text_feat" in encoder_outputs:
            feats.append(encoder_outputs["text_feat"])
        
        visual_feat = torch.cat(feats, dim=1)  # [B, N, C]
        # Forward pass through the ViT model
        b, n, c = visual_feat.shape
        query_embed = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, num_actions, hidden_dim]
        
        action_embedding = self.model(tgt=query_embed, memory=visual_feat)
        logits = self.action_decoder(action_embedding)  # [B, num_actions, action_space * 3]
        
        outputs = {
            "logits": logits,  # [B, num_actions, action_space]
        }

        labels = batch_data.get("action")  # Assuming labels are provided in batch_data under "actions"

        if labels is not None:
            # If labels are provided, compute the loss
            loss = self.criterion(logits.flatten(start_dim=1), labels.flatten(start_dim=1))
            outputs["total_loss"] = loss

            
            trans_loss = F.mse_loss(logits[:, :3].flatten(start_dim=1), labels[:, :3].flatten(start_dim=1))
            rot_loss = F.mse_loss(logits[:, 3:6].flatten(start_dim=1), labels[:, 3:6].flatten(start_dim=1))
            gripper_loss = F.mse_loss(logits[:, 6].flatten(start_dim=1), labels[:, 6].flatten(start_dim=1))
            l1_gripper_loss = F.l1_loss(logits[:, 6].flatten(start_dim=1), labels[:, 6].flatten(start_dim=1))
            
            outputs.update({
                "trans_loss": trans_loss,
                "rot_loss": rot_loss,
                "gripper_loss": gripper_loss,
                "l1_gripper_loss": l1_gripper_loss,
            })

        return outputs

if __name__ == "__main__":
    vit_tiny = TransformerAction()
    vit_tiny.eval()

    # Test with a variable image size (e.g., 128x128)
    img_size = 256
    input_channels = 5
    dummy_input = torch.randn(16, 5, img_size, img_size)  # 1 image, 5 channels, 128x128
    # output = vit_tiny(dummy_input)

    action_gt = torch.randn(16, 10, 7)  # Assuming 7 action dimensions
    output = vit_tiny(dummy_input, labels=action_gt)

    print(output["logits"].shape)  # Default output [1, 7] (for action vectors)
    if "loss" in output:
        print("Loss:", output["loss"].item())
    else:
        print("No loss computed.")