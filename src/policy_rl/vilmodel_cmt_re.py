import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, 
                                         None if head_mask is None else head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertOutAttention(nn.Module):
    """
    Cross attention: query from input_tensor, key/value from context
    用于: obs 作为 query, history 作为 key/value
    """
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        # query 来自输入, key/value 来自 context
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        """
        Args:
            hidden_states: query 的来源 (在这里是 obs/观测)
            context: key 和 value 的来源 (在这里是 history/历史)
            attention_mask: 注意力掩码
        """
        # 只有 hidden_states 产生 query
        mixed_query_layer = self.query(hidden_states)
        # context 产生 key 和 value
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply the attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, attention_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, attention_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores


class RECrossAttentionLayer(nn.Module):
    """
    交叉注意力层 for RE model:
    - obs (观测) 作为 query
    - history (历史) 作为 key/value
    - 单向注意力: obs attends to history
    """
    def __init__(self, config):
        super().__init__()
        
        # History 的自注意力层
        self.hist_self_att = BertAttention(config)
        self.hist_inter = BertIntermediate(config)
        self.hist_output = BertOutput(config)
        
        # 交叉注意力: obs attends to history
        # query 来自 obs, key/value 来自 history
        self.cross_attention = BertOutAttention(config, ctx_dim=config.hidden_size)
        self.cross_output = BertSelfOutput(config)
        
        # FFN 层
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def forward(self, hist_embeds, ob_embeds, extended_hist_masks, extended_ob_masks):
        """
        Args:
            hist_embeds: 历史嵌入 (batch, hist_len, hidden)
            ob_embeds: 观测嵌入 (batch, ob_len, hidden)
            extended_hist_masks: 历史注意力掩码
            extended_ob_masks: 观测注意力掩码
        """
        # 1. History 自注意力
        hist_att_output = self.hist_self_att(hist_embeds, extended_hist_masks)[0]
        hist_inter_output = self.hist_inter(hist_att_output)
        hist_output = self.hist_output(hist_inter_output, hist_att_output)
        
        # 2. 交叉注意力: obs attends to history
        # obs 作为 query, history 作为 key/value
        # cross_attention 内部: query 来自 hidden_states (ob), key/value 来自 context (history)
        ob_cross_output, _ = self.cross_attention(ob_embeds, hist_output, attention_mask=extended_hist_masks)
        ob_cross_output = self.cross_output(ob_cross_output, ob_embeds)
        
        # 3. Output FFN
        visn_inter_output = self.visn_inter(ob_cross_output)
        visn_output = self.visn_output(visn_inter_output, ob_cross_output)
        
        return visn_output


class REEncoder(nn.Module):
    """
    Modified Encoder for RE (Reverie) model:
    - 无 text/language 输入
    - 只有 history (历史) 和 obs (观测)
    - 交叉注意力: obs 作为 query, history 作为 key/value
    """
    def __init__(self, config):
        super().__init__()

        self.num_h_layers = config.num_h_layers
        self.num_r_layers = config.num_r_layers
        self.num_x_layers = config.num_x_layers

        # History 编码层
        self.h_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_h_layers)]
        ) if self.num_h_layers > 0 else None
        
        # Observation (图像) 编码层
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        ) if self.num_r_layers > 0 else None
        
        # 交叉注意力层: obs attends to history
        self.x_layers = nn.ModuleList(
            [RECrossAttentionLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, hist_embeds, ob_embeds, extended_hist_masks, extended_ob_masks):
        """
        Args:
            hist_embeds: 历史嵌入 (batch, hist_len, hidden)
            ob_embeds: 观测嵌入 (batch, ob_len, hidden)
            extended_hist_masks: 历史注意力掩码 (batch, 1, 1, hist_len)
            extended_ob_masks: 观测注意力掩码 (batch, 1, 1, ob_len)
        
        Returns:
            hist_embeds: 编码后的历史
            ob_embeds: 编码后的观测（包含来自历史的上下文信息）
        """
        # 1. History 编码
        if self.h_layers is not None:
            for layer_module in self.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]
        
        # 2. Observation 编码
        if self.r_layers is not None:
            for layer_module in self.r_layers:
                temp_output = layer_module(ob_embeds, extended_ob_masks)
                ob_embeds = temp_output[0]

        # 3. 交叉注意力: obs attends to history
        # obs 作为 query, history 作为 key/value
        for layer_module in self.x_layers:
            ob_embeds = layer_module(hist_embeds, ob_embeds, extended_hist_masks, extended_ob_masks)

        return hist_embeds, ob_embeds


class ImageEmbeddings(nn.Module):
    """观测/图像嵌入层"""
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 0: non-navigable, 1: navigable, 2: stop
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, ang_feat, type_embeddings, nav_types=None):
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
        embeddings = transformed_im + transformed_ang + type_embeddings
        if nav_types is not None:
            nav_embeddings = self.nav_type_embedding(nav_types)
            embeddings = embeddings + nav_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class HistoryEmbeddings(nn.Module):
    """历史嵌入层"""
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
        self.position_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        # 历史专用类型嵌入
        self.type_embedding = nn.Embedding(1, config.hidden_size)

        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hist_enc_pano = config.hist_enc_pano
        if config.hist_enc_pano:
            self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.pano_ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
            self.pano_ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            pano_enc_config = copy.copy(config)
            pano_enc_config.num_hidden_layers = config.num_h_pano_layers
            self.pano_encoder = BertEncoder(pano_enc_config)
        else:
            self.pano_encoder = None

    def forward(self, img_feats, ang_feats, pos_ids, 
                pano_img_feats=None, pano_ang_feats=None):
        '''Args:
        - img_feats: (batch_size, dim_feat) 当前视角的图像特征
        - pos_ids: (batch_size, ) 位置ID
        - pano_img_feats: (batch_size, pano_len, dim_feat) 全景特征
        '''
        device = next(iter(self.parameters())).device
        if img_feats is not None:
            batch_size = img_feats.size(0)
        else:
            # 从 pos_ids 获取 batch_size
            batch_size = pos_ids.size(0) if pos_ids is not None else 1

        type_ids = torch.zeros((batch_size, )).long().to(device)
        type_embeddings = self.type_embedding(type_ids)

        if img_feats is None:
            cls_embeddings = self.dropout(self.layer_norm(
                self.cls_token.expand(batch_size, -1, -1)[:, 0] + type_embeddings))
            return cls_embeddings

        # 历史嵌入 = 图像特征 + 角度特征 + 位置嵌入 + 类型嵌入
        embeddings = self.img_layer_norm(self.img_linear(img_feats)) + \
                     self.ang_layer_norm(self.ang_linear(ang_feats)) + \
                     self.position_embeddings(pos_ids) + \
                     type_embeddings

        if self.pano_encoder is not None:
            pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                              self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
            pano_embeddings = self.dropout(pano_embeddings)
            batch_size, pano_len, _ = pano_img_feats.size()
            extended_pano_masks = torch.zeros(batch_size, pano_len).float().to(device).unsqueeze(1).unsqueeze(2)
            pano_embeddings = self.pano_encoder(pano_embeddings, extended_pano_masks)[0]
            pano_embeddings = torch.mean(pano_embeddings, 1)
            embeddings = embeddings + pano_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NextActionPrediction(nn.Module):
    """动作预测头"""
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class NavCMTRE(BertPreTrainedModel):
    """
    Modified NavCMT for RE (Reverie) model:
    - 无 text/language 输入
    - 只有 history (历史) 和 obs (观测)
    - 交叉注意力: obs 作为 query, history 作为 key/value
    """
    def __init__(self, config):
        super().__init__(config)
        self.img_embeddings = ImageEmbeddings(config)
        self.hist_embeddings = HistoryEmbeddings(config)
        self.encoder = REEncoder(config)
        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        # self.init_weights()

    def forward(self, mode,
                hist_img_feats=None, hist_ang_feats=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
                ob_masks=None):
        
        # History 嵌入模式
        if mode == 'history':
            hist_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, ob_step_ids,
                pano_img_feats=hist_pano_img_feats, pano_ang_feats=hist_pano_ang_feats)
            if self.config.fix_hist_embedding:
                hist_embeds = hist_embeds.detach()
            return hist_embeds
            
        # Visual 模式: obs attends to history
        elif mode == 'visual':
            """
            extended_hist_masks 和 extended_hist_ob_masks 的作用:
            
            1. extended_hist_masks (历史注意力掩码):
               - 形状: (batch_size, 1, 1, hist_len)
               - 作用: 遮盖历史序列中的 padding 部分
               - 原始 hist_masks 是 (batch_size, hist_len) 的布尔值
               - 扩展后变成 4D 掩码用于注意力计算
            
            2. extended_ob_masks (观测注意力掩码):
               - 形状: (batch_size, 1, 1, ob_len)
               - 作用: 遮盖观测序列中的 padding 部分
               - 原始 ob_masks 是 (batch_size, ob_len) 的布尔值
            
            掩码转换过程:
            - 原始: hist_masks = tensor([True, True, False, False])  # True = 有效, False = padding
            - 扩展: extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0
              这样 True 变成 0, False 变成 -10000 (极大的负数)
              在 softmax 后，-10000 的位置注意力权重接近 0
            """
            
            # ==================== 处理历史特征 ====================
            # 将 hist_masks 扩展为 4D 注意力掩码
            # hist_masks: (batch_size, hist_len) -> (batch_size, 1, 1, hist_len)
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
            # 转换: True(有效) -> 0, False(padding) -> -10000
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

            # History 编码层： 这部分为对全部历史编码进行transformer： temporal transformer
            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(hist_embeds, extended_hist_masks)
                    hist_embeds = temp_output[0]

            # ==================== 处理观测特征 ====================
            # 将 ob_masks 扩展为 4D 注意力掩码
            # ob_masks: (batch_size, ob_len) -> (batch_size, 1, 1, ob_len)
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            # 转换: True(有效) -> 0, False(padding) -> -10000
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0

            # 观测嵌入
            ob_token_type_ids = torch.ones(ob_img_feats.size(0), ob_img_feats.size(1), dtype=torch.long, device=self.device)
            ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats, 
                self.img_embeddings.nav_type_embedding.weight[ob_token_type_ids.long()], 
                nav_types=ob_nav_types)
            
            # Observation 编码层
            if self.encoder.r_layers is not None:
                for layer_module in self.encoder.r_layers:
                    temp_output = layer_module(ob_embeds, extended_ob_masks)
                    ob_embeds = temp_output[0]
            if self.config.fix_obs_embedding:
                ob_embeds = ob_embeds.detach()

            # ==================== 交叉注意力: obs attends to history ====================
            # obs 作为 query, history 作为 key/value
            # 这让当前观测可以关注历史中的所有信息
            hist_embeds, ob_embeds = self.encoder(
                hist_embeds, ob_embeds, 
                extended_hist_masks, extended_ob_masks
            )

            # ==================== 动作预测 ====================
            # 基于观测特征预测动作（观测已通过交叉注意力获取了历史信息）
            act_logits = self.next_action(ob_embeds).squeeze(-1)
            act_logits.masked_fill_(ob_nav_types==0, -float('inf'))

            # 为了兼容，返回: act_logits, dummy_txt_embeds, hist_embeds, ob_embeds
            batch_size = ob_embeds.size(0)
            dummy_txt_embeds = torch.zeros(batch_size, 1, self.config.hidden_size, device=ob_embeds.device)
            
            return act_logits, dummy_txt_embeds, hist_embeds, ob_embeds
