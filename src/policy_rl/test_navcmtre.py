"""
测试 NavCMTRE 模型的输入输出
测试模型: vilmodel_cmt_re.py 中定义的 NavCMTRE
"""

import torch
import numpy as np
from transformers import PretrainedConfig


class Args:
    """模拟命令行参数"""
    # 数据集参数
    dataset = 'r2r'
    tokenizer = 'bert'
    bert_ckpt_file = None  # 不加载预训练权重
    
    # 特征参数
    image_feat_size = 768
    angle_feat_size = 128
    
    # 模型层数参数
    num_l_layers = 1  # 语言层数
    num_r_layers = 1   # 视觉层数 (观测编码)
    num_h_layers = 1  # 历史层数
    num_x_layers = 1  # 交叉注意力层数
    
    # 历史编码参数
    hist_enc_pano = True
    hist_pano_num_layers = 1
    
    # 嵌入固定参数
    fix_lang_embedding = False
    fix_hist_embedding = False
    fix_obs_embedding = False
    
    # 其他参数
    no_lang_ca = False
    act_pred_token = 'ob'


def create_config(args):
    """创建模型配置 (不下载任何预训练权重)"""
    # 手动创建配置，不使用 from_pretrained 下载权重
    vis_config = PretrainedConfig()
    
    # BERT 基本配置
    vis_config.vocab_size = 30522
    vis_config.hidden_size = 768
    vis_config.num_hidden_layers = 12
    vis_config.num_attention_heads = 12
    vis_config.intermediate_size = 3072
    vis_config.hidden_act = 'gelu'
    vis_config.hidden_dropout_prob = 0.1
    vis_config.attention_probs_dropout_prob = 0.1
    vis_config.max_position_embeddings = 512
    vis_config.type_vocab_size = 2
    vis_config.layer_norm_eps = 1e-12
    
    # 任务相关配置
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_r_layers = args.num_r_layers
    vis_config.num_h_layers = args.num_h_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.hist_enc_pano = args.hist_enc_pano
    vis_config.num_h_pano_layers = args.hist_pano_num_layers

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_hist_embedding = args.fix_hist_embedding
    vis_config.fix_obs_embedding = args.fix_obs_embedding

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1

    vis_config.no_lang_ca = args.no_lang_ca
    vis_config.act_pred_token = args.act_pred_token
    
    return vis_config


def test_model():
    """测试 NavCMTRE 模型"""
    print("=" * 60)
    print("测试 NavCMTRE 模型 (无文本输入, obs attends to history)")
    print("=" * 60)
    
    # 创建配置
    args = Args()
    config = create_config(args)
    
    # 导入模型
    from vilmodel_cmt_re import NavCMTRE
    
    # 创建模型
    print("\n[1] 创建模型...")
    model = NavCMTRE(config)
    model.eval()
    print(f"    模型创建成功!")
    print(f"    - History 编码层数: {config.num_h_layers}")
    print(f"    - Observation 编码层数: {config.num_r_layers}")
    print(f"    - 交叉注意力层数: {config.num_x_layers}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"    设备: {device}")
    
    # ==================== 测试参数 ====================
    batch_size = 2
    hist_len = 5       # 历史长度
    ob_len = 12         # 观测候选点数量
    image_feat_size = args.image_feat_size
    angle_feat_size = args.angle_feat_size
    hidden_size = config.hidden_size
    
    print(f"\n[2] 准备测试数据...")
    print(f"    Batch size: {batch_size}")
    print(f"    History 长度: {hist_len}")
    print(f"    Observation 长度: {ob_len}")
    print(f"    Image feature size: {image_feat_size}")
    print(f"    Angle feature size: {angle_feat_size}")
    print(f"    Hidden size: {hidden_size}")
    
    # 历史特征 (历史视角的图像特征)
    hist_img_feats = torch.randn(batch_size, image_feat_size).to(device)
    # 历史角度特征
    hist_ang_feats = torch.randn(batch_size, angle_feat_size).to(device)
    # 历史全景特征 (可选)
    hist_pano_img_feats = None
    hist_pano_ang_feats = None
    if args.hist_enc_pano:
        pano_len = 12
        hist_pano_img_feats = torch.randn(batch_size, pano_len, image_feat_size).to(device)
        hist_pano_ang_feats = torch.randn(batch_size, pano_len, angle_feat_size).to(device)
    
    # 观测图像特征 (候选点)
    ob_img_feats = torch.randn(batch_size, ob_len, image_feat_size).to(device)
    # 观测角度特征
    ob_ang_feats = torch.randn(batch_size, ob_len, angle_feat_size).to(device)
    # 观测导航类型 (0: 不可导航, 1: 可导航, 2: 停止)
    ob_nav_types = torch.randint(1, 3, (batch_size, ob_len)).to(device)
    # 观测有效掩码 (True = 有效)
    ob_masks = torch.ones(batch_size, ob_len, dtype=torch.bool).to(device)
    ob_masks[1, -2:] = False  # 第二个样本最后两个是padding
    
    # 历史有效掩码
    hist_masks = torch.ones(batch_size, hist_len, dtype=torch.bool).to(device)
    hist_masks[0, -1] = False  # 第一个样本最后一个是padding
    
    # 位置ID
    ob_step_ids = torch.tensor([0, 1]).to(device)
    
    print(f"\n[3] 测试 'history' 模式...")
    # 测试 history 模式
    with torch.no_grad():
        hist_embeds = model(
            mode='history',
            hist_img_feats=hist_img_feats,
            hist_ang_feats=hist_ang_feats,
            hist_pano_img_feats=hist_pano_img_feats,
            hist_pano_ang_feats=hist_pano_ang_feats,
            ob_step_ids=ob_step_ids
        )
    
    print(f"    历史嵌入输出形状: {hist_embeds.shape}")
    assert hist_embeds.shape == (batch_size, hidden_size), "历史嵌入形状错误!"
    print(f"    ✓ History 模式测试通过!")
    
    # 扩展历史嵌入以匹配历史长度
    # 在实际使用中，hist_embeds 是逐步累积的列表
    # 这里我们创建一个简化的测试
    hist_embeds_test = hist_embeds.unsqueeze(1).expand(batch_size, hist_len, hidden_size)
    hist_lens = [hist_len, hist_len]
    
    print(f"\n[4] 测试 'visual' 模式...")
    # 测试 visual 模式
    with torch.no_grad():
        outputs = model(
            mode='visual',
            hist_embeds=hist_embeds_test,
            hist_masks=hist_masks,
            ob_img_feats=ob_img_feats,
            ob_ang_feats=ob_ang_feats,
            ob_nav_types=ob_nav_types,
            ob_masks=ob_masks
        )
    
    act_logits, dummy_txt_embeds, hist_embeds_out, ob_embeds_out = outputs
    
    print(f"    动作logits形状: {act_logits.shape}")
    print(f"    (期望: ({batch_size}, {ob_len}))")
    assert act_logits.shape == (batch_size, ob_len), "动作logits形状错误!"
    
    print(f"    历史嵌入输出形状: {hist_embeds_out.shape}")
    print(f"    (期望: ({batch_size}, {hist_len}, {hidden_size}))")
    assert hist_embeds_out.shape == (batch_size, hist_len, hidden_size), "历史嵌入形状错误!"
    
    print(f"    观测嵌入输出形状: {ob_embeds_out.shape}")
    print(f"    (期望: ({batch_size}, {ob_len}, {hidden_size}))")
    assert ob_embeds_out.shape == (batch_size, ob_len, hidden_size), "观测嵌入形状错误!"
    
    print(f"    ✓ Visual 模式测试通过!")
    
    # ==================== 测试梯度流动 ====================
    print(f"\n[5] 测试梯度流动...")
    model.train()
    
    # 重新创建需要梯度的输入
    hist_img_feats.requires_grad = True
    hist_ang_feats.requires_grad = True
    ob_img_feats.requires_grad = True
    ob_ang_feats.requires_grad = True
    
    # History 模式
    hist_embeds = model(
        mode='history',
        hist_img_feats=hist_img_feats,
        hist_ang_feats=hist_ang_feats,
        hist_pano_img_feats=hist_pano_img_feats,
        hist_pano_ang_feats=hist_pano_ang_feats,
        ob_step_ids=ob_step_ids
    )
    
    hist_embeds_test = hist_embeds.unsqueeze(1).expand(batch_size, hist_len, hidden_size).detach()
    hist_embeds_test.requires_grad = True
    
    # Visual 模式
    outputs = model(
        mode='visual',
        hist_embeds=hist_embeds_test,
        hist_masks=hist_masks,
        ob_img_feats=ob_img_feats,
        ob_ang_feats=ob_ang_feats,
        ob_nav_types=ob_nav_types,
        ob_masks=ob_masks
    )
    
    act_logits = outputs[0]
    
    # 计算损失 (简单的方式: 取第一个样本第一个动作的logit)
    loss = act_logits[0, 0]
    loss.backward()
    
    print(f"    损失值: {loss.item():.4f}")
    print(f"    ✓ 梯度流动测试通过!")
    
    # ==================== 测试无历史情况 ====================
    print(f"\n[6] 测试初始状态 (无历史)...")
    model.eval()
    with torch.no_grad():
        # 初始历史嵌入 (只有 CLS token)
        init_hist_embeds = model(
            mode='history',
            hist_img_feats=None,
            hist_ang_feats=None,
            ob_step_ids=ob_step_ids
        )
    
    print(f"    初始历史嵌入形状: {init_hist_embeds.shape}")
    assert init_hist_embeds.shape == (batch_size, hidden_size), "初始历史嵌入形状错误!"
    print(f"    ✓ 初始状态测试通过!")
    
    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
    print("""
模型说明:
- NavCMTRE 是无文本输入的导航模型
- 输入: history (历史) + obs (观测)
- 交叉注意力机制: obs 作为 query, history 作为 key/value
- 输出: 动作 logits (用于预测下一步动作)

输入参数:
- hist_img_feats: 历史图像特征 (batch, image_feat_size)
- hist_ang_feats: 历史角度特征 (batch, angle_feat_size)
- ob_img_feats: 观测图像特征 (batch, ob_len, image_feat_size)
- ob_ang_feats: 观测角度特征 (batch, ob_len, angle_feat_size)
- ob_nav_types: 观测导航类型 (batch, ob_len)
- hist_masks: 历史有效掩码 (batch, hist_len)
- ob_masks: 观测有效掩码 (batch, ob_len)

输出:
- act_logits: 动作logits (batch, ob_len)
- hist_embeds: 编码后的历史 (batch, hist_len, hidden)
- ob_embeds: 编码后的观测 (batch, ob_len, hidden)
    """)


if __name__ == '__main__':
    test_model()
