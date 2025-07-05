
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SiglipVisionConfig:
    
    def __init__(
        self,
        embed_dim=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_patches: int = None,
        **kwargs
    ):
        pass
        super().__init__()
        self.embed_dim = embed_dim # size of embedding vectors
        self.intermediate_size = intermediate_size # size of linear layers in FFW
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_patches = num_patches # how many output embeddings does the ViT output (how many image embeddings we have for each image) -> number of patches
        
        

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )
        self.num_patches = (self.image_size // self.patch_size) **2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            # tensor of size (1,num_positions) with values from 0 to num_positions - 1
            # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,...,14^2]
            torch.arange(self.num_positions).expand((1,-1)),
            persistan=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _,_,height, width = pixel_values.shape # [B,C,H,W]
        # num_patch_H = H/patch_size (same for num_patch_W)
        patch_embeds = self.patch_embedding(pixel_values) # [B, embed_dim, num_patch_H, num_patch_W]
        embeddings = patch_embeds.flatten(2) # [B, embed_dim, num_patches=npH*npW]
        embeddings = patch_embeds.transpose(1,2) # [B, num_patches, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
        
        
class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # input: [B, NP, E]
        batch_size, num_patch, embed_dim = hidden_states.shape
        query_states = self.q_proj(hidden_states) # [B, NP, E]
        key_states = self.q_proj(hidden_states) # [B, NP, E]
        value_states = self.v_proj(hidden_states) # [B, NP, E]
        # Split into multiple heads
        # Purpose: each
        query_states = query_states.view(batch_size, num_patch, self.num_heads, self.head_dim) # [B, NP, NH, EH]
        query_states = query_states.transpose(1, 2) # [B, NH, NP, EH]
        key_states = key_states.view(batch_size, num_patch, self.num_heads, self.head_dim) # [B, NP, NH, EH]
        key_states = key_states.transpose(1, 2) # [B, NH, NP, EH]
        value_states = value_states.view(batch_size, num_patch, self.num_heads, self.head_dim) # [B, NP, NH, EH]
        value_states = value_states.transpose(1, 2) # [B, NH, NP, EH]
        
        # Q* K^T / sqrt(d_k)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale) # [B, NH, NP, NP]

        # raise some ValueError here
        
        # apply softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) # [B, NH, NP, NP]

        # we don't use dropout since paligemma does not use dropout ?
        attn_weights = nn.functional.dropout(attn_weights, p = self.dropout, training=self.training) # [B, NH, NP, NP]

        attn_output = torch.matmul(attn_weights, value_states) # [B, NH, NP, EH]

        # Merge all attention heads
        # contiguous() is use to reduce computational overhead
        attn_output = attn_output.transpose(1,2).contiguous() # [B, NP, NH, EH]
        attn_output = attn_output.reshape(batch_size, num_patch, self.embed_dim) # [B, NP, E]

        attn_output = self.out_proj(attn_output) # [B, NP, E]
        return attn_output, attn_weights
        
        
        
        

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # Made up of 2 layers + 1 non-linear transformation 
        self.fc1 = nn.Linear(config.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states : [batch_size, num_patches, embed_dim]
        hidden_states = self.fc1(hidden_states) # [batch_size, num_patches, intermediate_size]
        # in relu, if input is negative -> output is always zero => no gradient so it will not flow
        # leaky relu,... allow a little bit of gradient flow from the negative side
        # => nonlinearity tell how the gradient will flow during back propagation
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh") # [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc2(hidden_states) # [batch_size, num_patches, embed_dim]
        return hidden_states
        
        
        
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states # [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states) # [batch_size, num_patches, embed_dim]
        # learn other -> contextualized
        hidden_states, _ = self.self_attn(hidden_states=hidden_states) # [batch_size, num_patches, embed_dim]
        hidden_states = hidden_states + residual # [batch_size, num_patches, embed_dim] 
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states) # [batch_size, num_patches, embed_dim]
        # learn independently -> model has more degree of freedom to learn whatever it's trying to learn
        # include non-linearity 
        # second use: prepare patches for the next layer
        hidden_states = self.mlp(hidden_states) # [batch_size, num_patches, embed_dim]
        hidden_states = hidden_states + residual # [batch_size, num_patches, embed_dim]
        return hidden_states
    
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for i in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
        
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        # Extract patches from images (using convolutional layer), flatten patches and add positional encoding
        hidden_states = self.embeddings(pixel_values) # [batch_size, num_patches, embed_dim]
        last_hidden_state = self.encoder(inputs_embeds=hidden_states) 
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
    
    
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig, *args):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values) -> Tuple:
        # [batch_size, channels, height, width] -> [batch_size, num_patches = num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)