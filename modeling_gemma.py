import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class GemmaConfig():
    def __init__(
        self,
        vocab_size, 
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        # Group attention heads: we have different number of heads for key-query-value
        num_attention_heads, # number of heads for query
        num_key_value_heads, # number of heads for key-value
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        pass


class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000, # token corresponding to the placeholder image token
        vocab_size=257152,
        projection_dim=2048, # final dim that image features should be resized to before feeding to language model
        hidden_size=2048, # embedding size of language model
        pad_token_id=None,
        **kwargs
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size)**2
        pass

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens


    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        # for applying positional encoding for each token
        # because we use rotary so positional encoding will be applied just before the attention
        # not at the beginning like others
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        hidden_states = input_embeds # [batch_size, seq_len, hidden_size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        
        for decoder_layer in self.layers:
            # [batch_size, seq_len, hidden_size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )




# Whenever we see causalLM in huggingface, its a transformer model + language modeling head (linear layer that project embeddings into logits)
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # transformer model
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    def tie_weights(self):
        # share the weight of embeddings layer with logits layer because they behave the same (just opposite)
        self.lm_head.weight = self.model.embed_tokens.weight
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        output = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        hidden_states = output
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        return_data = {
            "logits": logits
        }
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data


# convert the size of the image features extracted from the vision encoder into language model's embedding size
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_features):
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states


# we call it "conditional generation" because we conditionalize the generation of text base on the image given as input
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        # convert the embedding size of vision encoder to embedding size of text token to concatenate together
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        

    
    # reusing the parameters of embedding layer (token position -> embedding) in the final linear layer (embedding -> vocab size)
    # because the embedding layer functions the same as linear layer, just opposite
    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, 
        image_features: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ):
        batch_size_image, num_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        # probably they have tried multiple variations of the model and we want the magnitude of the numbers to remain the same
        # -> if we want to double the embedding size of the image features, we want the magnitude of numbers to remain the same
        # -> that's why we scale them
        scaled_image_features = image_features / (self.config.hidden_size**0.5) # [batch_size, seq_len, hidden_size]

        # combine the embeddings of the image tokens, text tokens and mask out all padding tokens
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device)
        # some mask that will be useful for understanding which is the placeholder token, which is the text token, padding token
        # although we will not use padding token
        
        # [batch_size, seq_len] True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # [batch_size, seq_len] True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # [batch_size, seq_len] True for padding tokens (all False since we don't have padding tokens)
        pad_mask = input_ids == self.pad_token_id 
        
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        
        # Add the text embeddings
        # mechanism of torch.where: if 1st argument is true, take input from 2nd argument, else take input from 3rd argument
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # insert image embeddings. Behave like torch.where, but cannot use torch.where because the sequence length of 
        # scaled_image_features is not equal to sequence length of final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####
        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]
        
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous
            # This only works wen we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )
            
            # Add the head dimension
            # [batch_size, q_len, kv_len] -> [batch_size, num_heads_q, q_len, kv_len]
            causal_mask = causal_mask.unsqueeze(1)
        
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask==0), 1).to(device)
        
        return final_embedding, causal_mask, position_ids
 
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        # attention_mask is provided by tokenizer
        # because we do not use any padding, the attention_mask will be a series of 1
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # Convert text tokens into embeddings (except the image token)
        input_embeds = self.language_model.get_input_embeddings()(input_ids) # [batch_size, seq_len, hidden_size]
        # Extract image embeddings
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype)) # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim] 
        # resize image embedding to have same size of language model embedding
        image_features = self.multi_modal_projector(selected_image_feature) # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size]
        # Merge text embeddings and image embeddings
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )
        