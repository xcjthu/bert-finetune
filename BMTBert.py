# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import bmtrain as bmt
from model_center.layer.layernorm import LayerNorm
from model_center.layer import Embedding, Linear
from model_center.model.basemodel import BaseModel
from model_center.model.config import BertConfig
from model_center.layer.blocks import TransformerBlock
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class Encoder(torch.nn.Module):
    """ Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """
    def __init__(self, 
            num_layers : int,
            dim_model : int, 
            dim_ff : int,
            num_heads : int,
            dim_head : int,
            dtype : torch.dtype = torch.half,
            int8 : bool = False, 
            norm_init_var : float = 1.0,
            norm_bias : bool = False,
            norm_eps : float = 1e-5, 
            att_init_mean : float = 0.0, 
            att_init_std : float = 0.02,
            att_bias : bool = False,
            att_mask_value : float = float("-inf"),
            ffn_init_mean : float = 0.0, 
            ffn_init_std : float = 0.02,
            ffn_bias : bool = False,
            ffn_activate_fn : str = "gated_gelu",
            pos_bias_type : str = "none",
            post_layer_norm : bool = False,
            length_scale : bool = False,
            attn_scale : bool = False,
            dropout_p : float = 0,
            parallel_ffn : bool = False,
        ):

        super().__init__()
        
        self.num_layers = num_layers

        self.layers = bmt.TransformerBlockList([
            TransformerBlock(
                dim_model = dim_model, 
                dim_ff = dim_ff,
                num_heads = num_heads,
                dim_head = dim_head,
                is_decoder = False,
                dtype = dtype, 
                int8 = int8,
                norm_eps = norm_eps, 
                norm_init_var = norm_init_var,
                norm_bias = norm_bias,
                att_init_mean = att_init_mean, 
                att_init_std = att_init_std,
                att_bias = att_bias,
                att_mask_value = att_mask_value,
                ffn_init_mean = ffn_init_mean, 
                ffn_init_std = ffn_init_std,
                ffn_bias = ffn_bias,
                ffn_activate_fn = ffn_activate_fn,
                pos_bias_type = pos_bias_type,
                post_layer_norm = post_layer_norm,
                length_scale = length_scale,
                attn_scale = attn_scale,
                dropout_p = dropout_p,
                parallel_ffn = parallel_ffn,
            )
            for _ in range(num_layers)
        ])

        self.output_layernorm = LayerNorm(
                    dim_norm = dim_model, 
                    bias = norm_bias, 
                    dtype = dtype,
                    eps = norm_eps,
                    init_var = norm_init_var)

    def forward(self, hidden_states : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : torch.Tensor = None,
                      ):
        """
        Args:
            hidden-states (:obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of encoder, might be the embedding of a batch of sequences. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc, seq_enc)``): Avoid invalid areas to participate in the calculation 
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_enc, seq_enc)``) Provides position information to attention mechanism.  

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``: The encoder output. 

        """
        # (batch, seq_enc, dim_model)
        hidden_states = self.layers(hidden_states, attention_mask, position_bias, None, None, None)
        # (batch, seq_enc, dim_model)
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states



class BertPooler(torch.nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states[:, 0, :])
        pooled_output = self.activation(pooled_output)
        return pooled_output

        
class BertLMHead(torch.nn.Module):
    def __init__(self, dim_model, vocab_size, norm_eps):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.act_fn = torch.nn.functional.gelu
        self.layer_norm = LayerNorm(dim_model, eps=norm_eps)
        self.decoder = Linear(dim_model, vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


class Bert(BaseModel):

    _CONFIG_TYPE = BertConfig

    def __init__(self, config: BertConfig):
        super().__init__()

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size,
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_embedding = Embedding(
            vocab_size = config.position_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.token_type_embedding = Embedding(
            vocab_size = config.type_size,
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.embed_dropout = torch.nn.Dropout(config.dropout_p)

        self.encoder = Encoder(
            num_layers = config.num_layers,
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
            num_heads = config.num_heads,
            dim_head = config.dim_head,
            dtype = config.dtype, 
            int8 = config.int8,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, 
            att_init_std = config.att_init_std,
            att_bias = config.att_bias,
            att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, 
            ffn_init_std = config.ffn_init_std,
            ffn_bias = config.ffn_bias,
            ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale,
            attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
            post_layer_norm = config.post_layer_norm,
        )

        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head:
            self.cls_projection = Linear(
                dim_out = self.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )
        if not self.tied:
            self.lm_head = BertLMHead(
                dim_model = config.dim_model,
                vocab_size = config.vocab_size,
                norm_eps = config.norm_eps,
            )

        self.pooler = BertPooler(config.dim_model)

    def forward(self,
                input_ids=None,
                length=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None, #unused
                inputs_embeds=None,
                encoder_hidden_states=None, #unused
                encoder_attention_mask=None, #unused
                output_attentions=None, #unused
                output_hidden_states=None, #unused
                return_dict=True,
                return_logits = False,
    ):
        """ This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass.
            You can use it as a regular PyTorch Module.
            You can also select the data and data type that you want the model to return through changing the value of `return_dict` and `return_logits`.

        Args:
            input_ids (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            length (:obj:`torch.Tensor` of shape ``(batch)``): Length of input sequence before padding.  
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Used to avoid performing attention on padding token indices.
            token_type_ids(:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused. 
            position_ids(:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused.
            head_mask (:obj:`torch.Tensor` of shape ``(num_layers, num_heads)``): Unused.
            inputs_embeds (:obj:`torch.Tensor` of shape ``(batch, seq_length, dim_model)``): Embedding of the input. You can choose to directly pass the inputs embedding to control the way of embedding. 
            encoder_hidden_states(:obj:`torch.Tensor` of shape(batch, seq_length, dim_model)): Unused.
            encoder_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused. 
            output_attentions (:obj:`torch.Tensor` of shape ``(batch, num_heads, seq_length, seq_length)``): Unused.
            output_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_length, dim_model)``): Unused.
            return_dict (:obj:`bool`): Whether to return a BaseModelOutputWithPoolingAndCrossAttentions instead of just a tuple.
            return_logits (:obj:`bool`): Whether to return the prediction score for each token in vocabulary (before softmax).

        Return:
            BaseModelOutputWithPoolingAndCrossAttentions or tuple or torch.Tensor of shape (batch, seq_length, vocab_output_size) or (batch, seqlen, cls_head): The Bert output. Depended on the value of `return_dict` and `return_logits` 

        """
        assert input_ids is not None or inputs_embeds is not None

        if input_ids is not None:
            batch = input_ids.size(0)
            seq_length = input_ids.size(1)
            device = input_ids.device
        else:
            batch = inputs_embeds.size(0)
            seq_length = inputs_embeds.size(1)
            device = inputs_embeds.device

        with torch.no_grad():

            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
            else:
                attention_mask = torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < length[:, None]
            attention_mask = attention_mask.view(batch, seq_length, 1) & attention_mask.view(batch, 1, seq_length)

            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)

            if token_type_ids is None:
                token_type_ids = torch.zeros(seq_length, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)

        if inputs_embeds is None:
            hidden_states = self.input_embedding(input_ids.to(torch.int32))
        else:
            hidden_states = inputs_embeds
        position_embeds = self.position_embedding(position_ids.to(torch.int32))
        token_type_embeds = self.token_type_embedding(token_type_ids.to(torch.int32))
        hidden_states = hidden_states + token_type_embeds + position_embeds

        hidden_states = self.embed_dropout(hidden_states)

        hidden_states = self.encoder(hidden_states, attention_mask)

        if self.cls_head:
            logits = self.cls_projection(hidden_states)
        elif self.tied:
            logits = self.input_embedding.projection(hidden_states)
        elif not self.tied:
            logits = self.lm_head(hidden_states)

        if return_logits:
            return logits

        pooled_output = self.pooler(hidden_states)

        if not return_dict:
            return (hidden_states, pooled_output, None, None, None, None)
        else:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=hidden_states,
                pooler_output=pooled_output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )