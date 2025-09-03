import torch.nn as nn
from typing import Callable, Optional
import torch
from torch import Tensor
import clip


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features1: Optional[int] = None,
        hidden_features2: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        withdrop: bool = False,
        bias: bool = True,
        final_bias: bool = True
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features1 = hidden_features1 or in_features
        self.fc1 = nn.Linear(in_features, hidden_features1, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features1, hidden_features2, bias=bias)
        self.fc3 = nn.Linear(hidden_features2, out_features, bias=final_bias)
        self.drop = nn.Dropout(drop)
        self.withdrop = withdrop
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        if self.withdrop:
            x = self.drop(x)
        
        x = self.fc2(x)
        x = self.act(x)
        if self.withdrop:
            x = self.drop(x)
        
        x = self.fc3(x)
        if self.withdrop:
            x = self.drop(x)
        x = self.sigmoid(x)
        return x


class VQFModel(nn.Module):
    def __init__(self, device, in_dims=768*2, withdrop=False, drop=0.5, angle_interval=10, distance_len=11):
        super().__init__()
        
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=device) 
        ## dinov2
        self.dinov2 = torch.hub.load('/home/users/wpp/dinov2', 'dinov2_vitb14',source='local').to(device)  # base; 16patch

        self.angle_bin = int(360 / angle_interval)
        out_dims = self.angle_bin * distance_len
        self.mlp = Mlp(in_dims, 
                  int(in_dims / 2),
                  int(in_dims / 4),
                  out_dims,
                  drop=drop,
                  withdrop=withdrop,
                  ).to(device)
    def preprocess(self, x):
        return self.preprocess(x).to(self.device)
    
    def forward(self, x):
        with torch.no_grad():
            patch_embedding = self.clip_model.image_patch_embedding(x)
            embedding_clip = torch.mean(patch_embedding, dim=1)
            # embedding_clip = torch.max(patch_embedding, dim=1)

            ## dino
            features_dict = self.dinov2.forward_features(x)
            patch_embedding_dino = features_dict['x_norm_patchtokens']
            embedding_dino = torch.mean(patch_embedding_dino, dim=1)
            
            # concat
            embedding = torch.concat([embedding_clip, embedding_dino], dim=-1)
        
        # mlp
        output = self.mlp(embedding)
        output = output.reshape(x.shape[0], -1, self.angle_bin)
        return output
    
    def mean_absolute_error_loss(self, x, target):
        # 
        criterion = nn.L1Loss(reduction='mean')
        loss = criterion(x, target)
        return loss
