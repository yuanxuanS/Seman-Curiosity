import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
from src.finetune.dataset_utils import get_loader, SampleLoader
from PIL import Image
import argparse
import torch
from torch import nn
import math
import time
from torchvision.transforms import v2
import torchvision
import copy

# 新增: 从 vilmodel_cmt_re 导入必要的类和函数
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


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
        outputs = (layer_output,) + attention_outputs[1:]
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

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertOutAttention(nn.Module):
    """Cross attention: query from input_tensor, key/value from context"""
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
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
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
    """交叉注意力层: obs attends to history"""
    def __init__(self, config):
        super().__init__()
        
        # History 的自注意力层
        self.hist_self_att = BertAttention(config)
        self.hist_inter = BertIntermediate(config)
        self.hist_output = BertOutput(config)
        
        # 交叉注意力: obs attends to history
        self.cross_attention = BertOutAttention(config, ctx_dim=config.hidden_size)
        self.cross_output = BertSelfOutput(config)
        
        # FFN 层
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def forward(self, hist_embeds, ob_embeds, extended_hist_masks, extended_ob_masks):
        # 1. History 自注意力
        hist_att_output = self.hist_self_att(hist_embeds, extended_hist_masks)[0]
        hist_inter_output = self.hist_inter(hist_att_output)
        hist_output = self.hist_output(hist_inter_output, hist_att_output)
        
        # 2. 交叉注意力: obs attends to history
        ob_cross_output, _ = self.cross_attention(ob_embeds, hist_output, attention_mask=extended_hist_masks)
        ob_cross_output = self.cross_output(ob_cross_output, ob_embeds)
        
        # 3. Output FFN
        visn_inter_output = self.visn_inter(ob_cross_output)
        visn_output = self.visn_output(visn_inter_output, ob_cross_output)
        
        return visn_output


class REEncoder(nn.Module):
    """Modified Encoder: history 和 obs 交叉注意力"""
    def __init__(self, config):
        super().__init__()

        self.num_h_layers = config.num_h_layers
        self.num_r_layers = config.num_r_layers
        self.num_x_layers = config.num_x_layers

        self.h_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_h_layers)]
        ) if self.num_h_layers > 0 else None
        
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        ) if self.num_r_layers > 0 else None
        
        self.x_layers = nn.ModuleList(
            [RECrossAttentionLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, hist_embeds, ob_embeds, extended_hist_masks, extended_ob_masks):
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
        for layer_module in self.x_layers:
            ob_embeds = layer_module(hist_embeds, ob_embeds, extended_hist_masks, extended_ob_masks)

        return hist_embeds, ob_embeds


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

        self.hist_enc_pano = config.hist_enc_pano if hasattr(config, 'hist_enc_pano') else False
        if self.hist_enc_pano:
            self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.pano_ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
            self.pano_ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            pano_enc_config = copy.copy(config)
            pano_enc_config.num_hidden_layers = config.num_h_pano_layers if hasattr(config, 'num_h_pano_layers') else 2
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

        if self.pano_encoder is not None and pano_img_feats is not None:
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


def build_feature_extractor(model_name, device, checkpoint_file=None):
    """使用 timm 库加载 ResNet 或 ViT 模型"""
    
    checkpoint_file = '/home/wpp/Seman-Curiosity/models/vit_base/pytorch_model.bin'
    # 使用 timm 库创建模型 (支持 ResNet152, ViT 等)
    model = timm.create_model(model_name, pretrained=False).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    model.eval()

    # 图像预处理配置
    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms, device


def process_features(args, device):
    
    # 加载模型
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name, device,args.checkpoint_file)
    
    data_pth = "/home/wpp/Seman-Curiosity//exps/dump/test_test/episodes_data"
    # "outputs_asample/imgs/test5_env1/rgb_all_data"
    sampler = SampleLoader(data_pth, glbstep=True)
    # inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  
    
    
    fts = []
    rgb_types = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240',
                 'rgb_270', 'rgb_300', 'rgb_330']
    type = ["bbsgt", "depth"]
    type.extend(rgb_types)
    for e in range(5):
        images = []
        # for step in range(5, 6,1):
        step = 5
        sample_data = sampler.get_sample_multimodality(
        e, 1, step, type, 0)
        # 获取图像
        for rgb_ in rgb_types:
            image = np.array(sample_data[rgb_].data, copy=True) 
            image = Image.fromarray(image) 
            images.append(image)
                

        # 图像预处理：转换为张量、应用变换; 放缩为vit图输入大小224
        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
    
        # ========== 关键步骤：使用 ResNet/ViT 提取特征 ==========
        
        b_fts = model.forward_features(images)
        b_fts = b_fts.data
        fts.append(b_fts.unsqueeze(0))
    
    # 合并所有特征
    fts = torch.concat(fts, 0)
    return fts

def angle_feature(heading, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), ] * (angle_feat_size // 2),
        dtype=np.float32)
    
def get_point_angle_feature(angle_feat_size, baseViewId=0, ):
    feature = np.empty((12, angle_feat_size), np.float32)
    

    
    angle_interval = math.radians(30)
    # 3. 基础参考航向（通常是当前agent的朝向） [cite: 180]
    base_heading = (baseViewId % 12) * angle_interval
    
    for ix in range(12):

        # 相对当前状态的偏移量即为 ix * 30度 
        inferred_heading = base_heading + (ix * angle_interval)
        # 计算相对于基准航向的相对角度 [cite: 178]
        rel_heading = inferred_heading - base_heading
        

        # 5. 调用原始编码函数生成高维特征 [cite: 177]
        feature[ix, :] = angle_feature(rel_heading, angle_feat_size)
        
    return feature

def get_all_point_angle_feature(angle_feat_size,):
    baseViewId = 0
    ang_fts = get_point_angle_feature(
        angle_feat_size, baseViewId, 
        ) # 构
    return torch.tensor(ang_fts)

BertLayerNorm = torch.nn.LayerNorm

class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 0: non-navigable, 1: navigable, 2: stop
        # self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, ang_feat,):
        '''
        img_feat: env*12*dim_i
        ang_feat: env*12*dim_a
        '''
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
        embeddings = transformed_im + transformed_ang 
        # if nav_types is not None:
        #     nav_embeddings = self.nav_type_embedding(nav_types)
        #     embeddings = embeddings + nav_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    


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

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
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
        if attention_mask is not None:
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

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class panorama_model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        with torch.no_grad():
            model_name = 'vit_base_patch16_224'
            self.vit_model, img_transforms, device = build_feature_extractor(model_name, device, )
            self.img_transforms = v2.Compose([trans for trans in img_transforms.transforms 
                                              if not isinstance(trans, torchvision.transforms.transforms.ToTensor) ])
        self.img_embeddings = ImageEmbeddings(config)
        
        # 新增: 历史嵌入层和编码器
        self.hist_embeddings = HistoryEmbeddings(config)
        self.encoder = REEncoder(config)
        
        # 保留原始的 self-attention 用于单帧特征处理
        self.attention = BertAttention(config)
        
        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        # Policy linear layer
        self.policy_linear = nn.Linear(config.hidden_size, config.hidden_size // 2)
        # critic linear layer
        self.critic_linear = nn.Linear(config.hidden_size // 2, 1)

        self.hidden_size = config.hidden_size
        
    @property
    def output_size(self):
        return self.hidden_size // 2
    
    def encoding(self, images):
        '''
        images: env*c*w*h
        '''
        fts = []
        for e in range(images.shape[0]):
            # t1 = time.time()
            images_ = images[e, ...]
            # images_l = []
            # for i in range(images_.shape[0]):
            #     images_l.append(Image.fromarray(images_[i, ...].cpu().numpy().astype(np.uint8)) )
            
            # images_ = torch.stack([self.img_transforms(image).to(self.device) for image in images_l], 0)
            images_ = self.img_transforms(images_.permute(0,3,1,2))
            
            # print(f"in encoder, type convert{time.time()- t1}")
            t2 = time.time()
            b_fts = self.vit_model.forward_features(images_)
            b_fts = b_fts.data
            fts.append(b_fts.unsqueeze(0))
            # print(f"in encoder, forward feature {time.time()- t2}")
        
        fts = torch.concat(fts, 0)
        return fts
    
    def forward(self, 
            obs,
            hist_embeds=None,
            hist_masks=None,
            compute_hist_embed=False,
            ):
        """
        Forward 函数:
        - 输入已经编码好的 hist_embeds（经过 history_embedding 编码后的嵌入）
        - hist_embeds: (env, hist_len, hidden_size) 或 None
        - hist_masks: (env, hist_len) 或 None
        
        Args:
            obs: 当前观测 (env, 12, H, W, C) - 全景图
            hist_embeds: 历史嵌入 (env, hist_len, hidden_size) - 已经过 history_embedding 编码
            hist_masks: 历史掩码 (env, hist_len) 或 None
            compute_hist_embed: 是否计算并返回当前观测的嵌入用于历史存储
        """
        env = obs.shape[0]
        
        # ========== 处理当前观测 ==========
        with torch.no_grad():
            ob_img_feats = self.encoding(obs).to(self.device)
            ang_feats = get_all_point_angle_feature(self.config.angle_feat_size, )
            ob_ang_feats = (ang_feats.unsqueeze(0)).repeat(env, 1,1).to(self.device)
        
        # 当前观测嵌入
        ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats)
        
        # ========== 处理历史嵌入 ==========
        if hist_embeds is not None:
            # hist_embeds: (env, hist_len, hidden_size)
            hist_len = hist_embeds.size(1)
            
            # 创建历史掩码（如果未提供）
            if hist_masks is None:
                hist_masks = torch.ones(env, hist_len, dtype=torch.bool, device=self.device)
            
            # 扩展掩码为 4D
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.config.dtype if hasattr(self.config, 'dtype') else torch.float32)
            extended_hist_masks = (1.0 - extended_hist_masks.float()) * -10000.0
            
            # 当前观测的掩码（全1，表示都有效）
            ob_len = ob_embeds.size(1)
            extended_ob_masks = torch.ones(env, 1, 1, ob_len, device=self.device, dtype=self.config.dtype if hasattr(self.config, 'dtype') else torch.float32)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0
            
            # 使用交叉注意力编码器
            hist_embeds_out, ob_embeds = self.encoder(
                hist_embeds, ob_embeds,
                extended_hist_masks, extended_ob_masks
            )
            
            # 聚合观测特征（使用注意力输出）
            attention_outputs = ob_embeds.sum(-2)
        else:
            # 无历史时（t=0），使用原始的 self-attention
            # 注意：cls_token 已经在 HistoryEmbeddings 中处理
            attention_outputs = self.attention(ob_embeds,)[0].sum(-2)

        # policy
        x = nn.ReLU()(self.policy_linear(attention_outputs))
        value = self.critic_linear(x).squeeze(-1)
        
        # 如果需要返回历史嵌入用于存储
        if compute_hist_embed:
            # 返回当前观测的嵌入（经过 img_embeddings 之后的）
            # ob_embeds: (env, 12, hidden_size)
            return value, x, ob_embeds
        
        return value, x
    
    def encode_history(self, hist_pano_img_feats, hist_img_feats, hist_masks):
        """
        将历史特征编码为历史嵌入
        
        Args:
            hist_pano_img_feats: 历史全景视角特征 (env, hist_len, views, image_feat_size)
            hist_img_feats: 历史action对应视角特征 (env, hist_len, image_feat_size)
            hist_masks: 历史掩码 (env, hist_len)
        
        Returns:
            hist_embeds: 历史嵌入 (env, hist_len, hidden_size)
        """
        env, hist_len, views, _ = hist_pano_img_feats.shape
        
        # 创建位置编码: [0, 1, 2, ..., hist_len-1]
        pos_ids = torch.arange(hist_len, dtype=torch.long, device=hist_pano_img_feats.device)
        pos_ids = pos_ids.unsqueeze(0).expand(env, -1)  # (env, hist_len)
        
        # 处理全景视角特征: (env, hist_len, views, image_feat_size) -> (env*hist_len, views, image_feat_size)
        # 对每个hist_len位置的全景视角做平均，得到该位置的特征
        pano_feats = hist_pano_img_feats.mean(dim=2)  # (env, hist_len, image_feat_size)
        
        # 处理action对应视角特征: 已经是 (env, hist_len, image_feat_size)
        
        # 角度特征（可以设为0或使用默认角度）
        ang_feats = torch.zeros(env, hist_len, self.config.angle_feat_size, device=hist_pano_img_feats.device)
        
        # 使用 history_embeddings 编码
        hist_embeds = self.hist_embeddings(
            pano_feats,  # 使用全景视角的平均特征
            ang_feats,
            pos_ids
        )
        
        # 移除 cls_token（如果存在）
        if hist_embeds.size(1) > hist_len:
            hist_embeds = hist_embeds[:, 1:, :]  # 移除 cls_token
        
        return hist_embeds
    
    def select_action_feats(self, pano_img_feats, actions):
        """
        根据actions选择对应步的视角特征作为hist_img_feats
        
        Args:
            pano_img_feats: 全景视角特征 (env, hist_len, views, image_feat_size)
            actions: 历史动作 (env, hist_len) - 每个元素是0-11的视角索引
        
        Returns:
            selected_feats: 选择的特征 (env, hist_len, image_feat_size)
        """
        env, hist_len, views, feat_size = pano_img_feats.shape
        # actions: (env, hist_len) -> (env, hist_len, 1)
        actions = actions.long().unsqueeze(-1)
        # 使用 gather 选择对应视角的特征
        # pano_img_feats: (env, hist_len, views, feat_size) -> (env, hist_len, 1, views) -> (env, hist_len, 1, feat_size)
        selected_feats = torch.gather(pano_img_feats, 2, actions.unsqueeze(-1).expand(-1, -1, 1, feat_size))
        selected_feats = selected_feats.squeeze(2)  # (env, hist_len, feat_size)
        return selected_feats
    
    def forward_with_history(self, 
            curr_pano_img_feats,
            curr_pano_ang_feats,
            hist_pano_img_feats=None,
            hist_pano_ang_feats=None,
            hist_actions=None,
            hist_masks=None,
            compute_hist_embed=False):
        """
        新的 forward 方法：接收历史全景特征和当前全景特征进行推理
        
        Args:
            curr_pano_img_feats: 当前步全景图像特征 (env, views, image_feat_size)
            curr_pano_ang_feats: 当前步全景角度特征 (env, views, angle_feat_size)
            hist_pano_img_feats: 历史全景图像特征 (env, hist_len, views, image_feat_size) 或 None
            hist_pano_ang_feats: 历史全景角度特征 (env, hist_len, views, angle_feat_size) 或 None
            hist_actions: 历史动作 (env, hist_len) 或 None
            hist_masks: 历史掩码 (env, hist_len) 或 None
            compute_hist_embed: 是否返回当前观测的嵌入用于历史存储
            
        Returns:
            value: 价值估计
            actor_features: 策略特征
            (可选) curr_img_embed: 当前观测的嵌入用于历史存储
        """
        env = curr_pano_img_feats.shape[0]
        
        # ========== 处理当前观测 ==========
        # 当前观测嵌入
        curr_embeds = self.img_embeddings(curr_pano_img_feats, curr_pano_ang_feats)
        
        # ========== 处理历史信息 ==========
        if hist_pano_img_feats is not None and hist_actions is not None:
            # hist_pano_img_feats: (env, hist_len, views, image_feat_size)
            # hist_actions: (env, hist_len)
            # 根据 action 选择对应步的视角特征作为 hist_img_feats
            hist_img_feats = self.select_action_feats(hist_pano_img_feats, hist_actions)  # (env, hist_len, image_feat_size)
            
            # 对历史全景特征取平均得到全景嵌入
            hist_pano_avg = hist_pano_img_feats.mean(dim=2)  # (env, hist_len, image_feat_size)
            
            # 根据 hist_actions 选择对应的角度特征
            hist_ang_feats = self.select_action_feats(hist_pano_ang_feats, hist_actions)  # (env, hist_len, angle_feat_size)
            
            # 创建位置编码
            hist_len = hist_pano_img_feats.size(1)
            pos_ids = torch.arange(hist_len, dtype=torch.long, device=hist_pano_img_feats.device)
            pos_ids = pos_ids.unsqueeze(0).expand(env, -1)  # (env, hist_len)
            
            # 使用 hist_img_feats 和 hist_ang_feats 编码历史，同时传入全景特征
            # 更新后的 HistoryEmbeddings 支持 pano_img_feats 和 pano_ang_feats 参数
            hist_embeds = self.hist_embeddings(
                hist_img_feats,
                hist_ang_feats,
                pos_ids,
                pano_img_feats=hist_pano_img_feats,  # 传入完整全景特征
                pano_ang_feats=hist_pano_ang_feats   # 传入完整全景角度特征
            )
            
            # 移除 cls_token（如果存在）
            if hist_embeds.size(1) > hist_len:
                hist_embeds = hist_embeds[:, 1:, :]  # 移除 cls_token
            
            # 创建历史掩码（如果未提供）
            if hist_masks is None:
                hist_masks = torch.ones(env, hist_len, dtype=torch.bool, device=curr_pano_img_feats.device)
            
            # 扩展掩码为 4D
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.config.dtype if hasattr(self.config, 'dtype') else torch.float32)
            extended_hist_masks = (1.0 - extended_hist_masks.float()) * -10000.0
            
            # 当前观测的掩码（全1，表示都有效）
            ob_len = curr_embeds.size(1)
            extended_ob_masks = torch.ones(env, 1, 1, ob_len, device=curr_embeds.device, dtype=self.config.dtype if hasattr(self.config, 'dtype') else torch.float32)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0
            
            # 使用交叉注意力编码器
            _, curr_embeds = self.encoder(
                hist_embeds, curr_embeds,
                extended_hist_masks, extended_ob_masks
            )
            
            # 聚合观测特征（使用注意力输出）
            attention_outputs = curr_embeds.sum(-2)
        else:
            # 无历史时（t=0），使用 history_embedding 的 cls_token 编码
            # 调用 hist_embeddings 获取 cls_token embeddings: (env, hidden_size)
            hist_embeds = self.hist_embeddings(None, None, None)
            # 扩展为 (env, 1, hidden_size) 匹配交叉注意力输入
            hist_embeds = hist_embeds.unsqueeze(1)  # (env, 1, hidden_size)
            
            # 创建历史掩码（全1，表示有效）- 只有一个虚拟历史token
            extended_hist_masks = torch.zeros(env, 1, 1, 1, device=curr_embeds.device, 
                                             dtype=self.config.dtype if hasattr(self.config, 'dtype') else torch.float32)
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0
            
            # 当前观测的掩码（全1，表示都有效）
            ob_len = curr_embeds.size(1)
            extended_ob_masks = torch.ones(env, 1, 1, ob_len, device=curr_embeds.device, 
                                          dtype=self.config.dtype if hasattr(self.config, 'dtype') else torch.float32)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0
            
            # 使用交叉注意力编码器
            _, curr_embeds = self.encoder(
                hist_embeds, curr_embeds,
                extended_hist_masks, extended_ob_masks
            )
            
            # 聚合观测特征（使用注意力输出）
            attention_outputs = curr_embeds.sum(-2)

        # policy
        x = nn.ReLU()(self.policy_linear(attention_outputs))
        value = self.critic_linear(x).squeeze(-1)
        
        # 如果需要返回历史嵌入用于存储
        if compute_hist_embed:
            # 返回当前观测的嵌入（经过 img_embeddings 之后的）
            return value, x, curr_embeds
        
        return value, x
                                         
class ModelConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model_config = {
    "pred_head_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    # "finetuning_task": null,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "image_feat_size": 768,
    "angle_feat_size": 2,
    # "image_prob_size": 1000,
    # "img_feature_type": "imagenet",
    # "initializer_range": 0.02,
    "intermediate_size": 3072,
    # "num_l_layers": 9,
    "num_r_layers": 2,
    "num_h_layers": 1,
    "num_x_layers": 2,
    # "num_h_pano_layers": 2,
    "layer_norm_eps": 1e-12,
    # "max_position_embeddings": 514,
    "max_action_steps": 100,
    "num_attention_heads": 12,
    # "num_hidden_layers": 12,
    # "num_labels": 2,
    "output_attentions": False,
    "output_hidden_states": False,
    # "pruned_heads": {},
    # "torchscript": False,
    # "type_vocab_size": 2,
    # "update_lang_bert": True,
    # "vocab_size": 250002,
    # "lang_bert_name": "xlm-roberta-base"
    }

