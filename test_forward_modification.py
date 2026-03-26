#!/usr/bin/env python3
"""
测试 forward_with_history 方法的修改是否合理
"""
import torch
import sys
sys.path.insert(0, '.')

from src.policy_rl.panorama_model import panorama_model, ModelConfig

def test_forward_with_history():
    """测试修改后的 forward_with_history 方法"""
    print("测试 forward_with_history 方法修改...")
    
    # 创建配置
    config_dict = {
        "pred_head_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "image_feat_size": 768,
        "angle_feat_size": 2,
        "intermediate_size": 3072,
        "num_r_layers": 2,
        "num_h_layers": 1,
        "num_x_layers": 2,
        "layer_norm_eps": 1e-12,
        "max_action_steps": 100,
        "num_attention_heads": 12,
        "output_attentions": False,
        "output_hidden_states": False,
    }
    
    config = ModelConfig(**config_dict)
    device = torch.device("cpu")
    
    # 创建模型
    model = panorama_model(config, device)
    model.eval()
    
    # 测试数据
    batch_size = 2
    views = 12
    hidden_size = config.hidden_size
    image_feat_size = config.image_feat_size
    angle_feat_size = config.angle_feat_size
    
    # 当前观测特征
    curr_pano_img_feats = torch.randn(batch_size, views, image_feat_size)
    curr_pano_ang_feats = torch.randn(batch_size, views, angle_feat_size)
    
    print(f"输入形状:")
    print(f"  curr_pano_img_feats: {curr_pano_img_feats.shape}")
    print(f"  curr_pano_ang_feats: {curr_pano_ang_feats.shape}")
    
    # 测试1: 无历史的情况
    print("\n测试1: 无历史输入")
    with torch.no_grad():
        result = model.forward_with_history(
            curr_pano_img_feats=curr_pano_img_feats,
            curr_pano_ang_feats=curr_pano_ang_feats,
            hist_pano_img_feats=None,
            hist_pano_ang_feats=None,
            hist_actions=None,
            hist_masks=None,
            compute_hist_embed=False
        )
    
    print(f"返回值类型: {type(result)}")
    print(f"返回值长度: {len(result)}")
    
    value, act_logits = result
    print(f"  value shape: {value.shape}")
    print(f"  act_logits shape: {act_logits.shape}")
    
    # 验证形状
    assert value.shape == (batch_size,), f"期望value形状({batch_size},), 得到{value.shape}"
    assert act_logits.shape == (batch_size, views), f"期望act_logits形状({batch_size}, {views}), 得到{act_logits.shape}"
    
    # 测试2: 有历史的情况
    print("\n测试2: 有历史输入")
    hist_len = 5
    hist_pano_img_feats = torch.randn(batch_size, hist_len, views, image_feat_size)
    hist_pano_ang_feats = torch.randn(batch_size, hist_len, views, angle_feat_size)
    hist_actions = torch.randint(0, views, (batch_size, hist_len))
    hist_masks = torch.ones(batch_size, hist_len, dtype=torch.bool)
    
    with torch.no_grad():
        result = model.forward_with_history(
            curr_pano_img_feats=curr_pano_img_feats,
            curr_pano_ang_feats=curr_pano_ang_feats,
            hist_pano_img_feats=hist_pano_img_feats,
            hist_pano_ang_feats=hist_pano_ang_feats,
            hist_actions=hist_actions,
            hist_masks=hist_masks,
            compute_hist_embed=False
        )
    
    value, act_logits = result
    print(f"  value shape: {value.shape}")
    print(f"  act_logits shape: {act_logits.shape}")
    
    assert value.shape == (batch_size,), f"期望value形状({batch_size},), 得到{value.shape}"
    assert act_logits.shape == (batch_size, views), f"期望act_logits形状({batch_size}, {views}), 得到{act_logits.shape}"
    
    # 测试3: 返回历史嵌入
    print("\n测试3: 返回历史嵌入")
    with torch.no_grad():
        result = model.forward_with_history(
            curr_pano_img_feats=curr_pano_img_feats,
            curr_pano_ang_feats=curr_pano_ang_feats,
            hist_pano_img_feats=hist_pano_img_feats,
            hist_pano_ang_feats=hist_pano_ang_feats,
            hist_actions=hist_actions,
            hist_masks=hist_masks,
            compute_hist_embed=True
        )
    
    print(f"返回值长度: {len(result)}")
    value, act_logits, curr_embeds = result
    print(f"  value shape: {value.shape}")
    print(f"  act_logits shape: {act_logits.shape}")
    print(f"  curr_embeds shape: {curr_embeds.shape}")
    
    assert value.shape == (batch_size,), f"期望value形状({batch_size},), 得到{value.shape}"
    assert act_logits.shape == (batch_size, views), f"期望act_logits形状({batch_size}, {views}), 得到{act_logits.shape}"
    assert curr_embeds.shape == (batch_size, views, hidden_size), f"期望curr_embeds形状({batch_size}, {views}, {hidden_size}), 得到{curr_embeds.shape}"
    
    print("\n✅ 所有测试通过!")
    print("\n总结:")
    print("- forward_with_history 现在返回 (value, act_logits) 或 (value, act_logits, curr_embeds)")
    print("- act_logits 形状为 (batch_size, 12)，表示12个视图的动作logits")
    print("- value 形状为 (batch_size,)，表示价值估计")
    
    return True

if __name__ == "__main__":
    try:
        test_forward_with_history()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)