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

def build_feature_extractor(model_name, device, checkpoint_file=None):
    """使用 timm 库加载 ResNet 或 ViT 模型"""
    
    checkpoint_file = '/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/models/vit_base/pytorch_model.bin'
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
    
    data_pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/test_test/episodes_data"
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
            self.vit_model, self.img_transforms, device = build_feature_extractor(model_name, device, )
    
        self.img_embeddings = ImageEmbeddings(config)
        self.attention = BertAttention(config)
        # self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        # Policy linear layer
        self.policy_linear = nn.Linear(config.hidden_size, config.hidden_size // 2)
        # critic linear layer
        self.critic_linear = nn.Linear(config.hidden_size // 2, 1)

        self.hidden_size = config.hidden_size
        
    @property
    def output_size(self):
        return self.hidden_size // 2
    
    def encoder(self, images):
        '''
        images: env*c*w*h
        '''
        fts = []
        for e in range(images.shape[0]):
            images_ = images[e, ...]
            images_l = []
            for i in range(images_.shape[0]):
                images_l.append(Image.fromarray(images_[i, ...].cpu().numpy().astype(np.uint8)) )
            images_ = torch.stack([self.img_transforms(image).to(self.device) for image in images_l], 0)
            b_fts = self.vit_model.forward_features(images_)
            b_fts = b_fts.data
            fts.append(b_fts.unsqueeze(0))
        
        fts = torch.concat(fts, 0)
        return fts
    
    def forward(self, 
            obs,
            ):
        with torch.no_grad():
            # 特征提取
            env = obs.shape[0]
            ob_img_feats = self.encoder(obs).to(self.device)
            ang_feats = get_all_point_angle_feature(self.config.angle_feat_size, )
            ob_ang_feats = (ang_feats.unsqueeze(0)).repeat(env, 1,1).to(self.device)
    
        # policy
        ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats)
        attention_outputs = self.attention(ob_embeds,)[0].sum(-2)
    #  act_logits = self.next_action(attention_outputs).squeeze(-1)

        x = nn.ReLU()(self.policy_linear(attention_outputs))
        return self.critic_linear(x).squeeze(-1), x
                                         
class ModelConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model_config = {
    "pred_head_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    # "finetuning_task": null,
    # "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "image_feat_size": 768,
    "angle_feat_size": 2,
    # "image_prob_size": 1000,
    # "img_feature_type": "imagenet",
    # "initializer_range": 0.02,
    # "intermediate_size": 3072,
    # "num_l_layers": 9,
    # "num_r_layers": 0,
    # "num_h_layers": 0,
    # "num_x_layers": 4,
    # "num_h_pano_layers": 2,
    "layer_norm_eps": 1e-12,
    # "max_position_embeddings": 514,
    # "max_action_steps": 100,
    "num_attention_heads": 12,
    # "num_hidden_layers": 12,
    # "num_labels": 2,
    "output_attentions": False,
    # "output_hidden_states": false,
    # "pruned_heads": {},
    # "torchscript": false,
    # "type_vocab_size": 2,
    # "update_lang_bert": true,
    # "vocab_size": 250002,
    # "lang_bert_name": "xlm-roberta-base"
    }

