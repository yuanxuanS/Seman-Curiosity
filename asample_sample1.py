from src.finetune.dataset_utils import get_loader, SampleLoader
from src.policy_rl.agents.utils.detect_utils import box_iou_calc
import numpy as np
from PIL import Image
import clip
import torch
from asample.constants import target_coco_categories
import pickle
import time

class ObjectTrack:
    """代表一个检出的独立物体轨迹"""
    def __init__(self, start_frame, obj_data):
        # history 存储每一帧的数据 {frame_idx: {box, label, score}}
        self.history = {start_frame: obj_data}
        self.frames = [start_frame]
        self.scores = 0.
        self.score_first = 0.
        self.score_last = 0.
        self.aggre_scores = {}      # 聚合所有重叠的frame的分数
        self.best_frames = {}
        self.is_active = True
        
    def has_object(self):
        return list(self.history.values())[0] is not None
    
    def get_frame_score(self, f):
        assert f in self.frames, f"frame {f} not in track"
        if f == min(self.frames): score = self.scores + self.score_first
        elif f == max(self.frames): score = self.scores + self.score_last
        else: score = self.scores
        return score 
        
def group_by_object_and_score(sequence_detections, ):
    """
    基于物体的生命周期进行分组评分
    """
    num_frames = len(sequence_detections)
    all_tracks = []
    active_tracks = []

    # --- 阶段 1: 轨迹关联 (把框归类到物体) ---
    for t in range(num_frames):
        curr_objs = sequence_detections[t].data
        matched_indices = set()
        matched_curr = False
        
        # 尝试将当前帧的框匹配到已有的活跃轨迹中
        for track in active_tracks:
            last_f = track.frames[-1]
            last_data = track.history[last_f]
            
            if last_data is None:
                if len(curr_objs) == 0:
                    track.history[t] = None
                    track.frames.append(t)
                    matched_curr = True
                else:
                    track.is_active = False
                break       # 
            else:
                if len(curr_objs) == 0: # 当前帧无任何目标，均创建新track
                    track.is_active = False
                    break
                
                best_iou = 0
                best_idx = -1
                for i in range(len(curr_objs)):
                    obj = curr_objs[i]
                    if i in matched_indices: continue
                    
                    curr_box = obj.pred_boxes.tensor.cpu().numpy()
                    last_box = last_data.pred_boxes.tensor.cpu().numpy()
                    iou = box_iou_calc(last_box, curr_box)
                    if iou > iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                if best_idx != -1:
                    track.history[t] = curr_objs[best_idx]
                    track.frames.append(t)
                    matched_indices.add(best_idx)
                else:
                    track.is_active = False # 这一帧没匹配上，认为轨迹结束
        
        if len(curr_objs) == 0:
            if matched_curr == False:
                new_track = ObjectTrack(t, None)
                all_tracks.append(new_track)
                active_tracks.append(new_track)
            else:
                continue
        else:
            # 为新出现的框创建新轨迹
            for i in range(len(curr_objs)):
                obj = curr_objs[i]
                if i not in matched_indices:
                    new_track = ObjectTrack(t, obj)
                    all_tracks.append(new_track)
                    active_tracks.append(new_track)
            
        matched_curr = False
        # 移除已失效的轨迹
        active_tracks = [tr for tr in active_tracks if tr.is_active]
    return all_tracks

def score_tracks(all_tracks, sequence_detections):
    num_frames = len(sequence_detections)
    # frame_values = np.zeros(num_frames)
    
    # def add_each_frame(track, score):
    #     for pt in track:
    #         for f in pt.frames:
    #             frame_values[f] += score
    
    def match_box(tracks, tgt_track,found_bool, check_prev=False, another_frame_idx=None):
        for pt in tracks:
            if not pt.has_object():
                continue
            
            another_box = pt.history[another_frame_idx].pred_boxes.tensor.cpu().numpy()
            curr_box = tgt_track.history[f1].pred_boxes.tensor.cpu().numpy()
            if box_iou_calc(another_box, curr_box) > iou_threshold:
                if pt.history[another_frame_idx].pred_classes != curr_label:
                    # 规则：类别变化 -> f1相邻前一帧 +1, f1 +1
                    # frame_values[prev_frame_idx] += 1
                    if check_prev: 
                        pt.score_last += 1
                        tgt_track.score_first += 1
                    else: 
                        pt.score_first += 1
                        tgt_track.score_last += 1
                    # frame_values[f1] += 1
                    found_bool = True
                    break # 找到一个匹配的翻转即可
        return found_bool
        
    for i, track in enumerate(all_tracks):
        if not track.has_object():
            continue
        
        f_indices = sorted(track.frames)
        f1 = f_indices[0]      # 当前 track 的第一帧
        fn = f_indices[-1]     # 当前 track 的最后一帧
        curr_label = track.history[f1].pred_classes
        
        # --- 1. 检查开头 (目标出现逻辑) ---
        if f1 > 0:
            prev_frame_idx = f1 - 1
            # 查找在 f1 前一帧活跃的其他 track (如果有的话)
            prev_tracks = [t for t in all_tracks if prev_frame_idx in t.frames]
            
            # 逻辑 A: 检查是否存在位置重合但类别不同的 track (类别变化)
            # 我们通过 IoU 判定 prev_frame 中的某个物体是否就是当前物体的“前身”
            found_class_flip = False
            found_class_flip = match_box(prev_tracks, track, found_class_flip, 
                                         check_prev=True,
                                         another_frame_idx=prev_frame_idx)
                    
            # 逻辑 B: 如果没有类别变化的 track，则视为纯粹的“目标出现”
            if not found_class_flip:
                # 规则：前一帧无目标(该实例) -> 前面所有帧(last track区间) +1, f1 +1
                # add_each_frame(prev_tracks, 0.5)
                for pt in prev_tracks:
                    pt.scores += 1
                track.scores += 1
        # --- 2. 检查结尾 (目标消失逻辑) ---
        if fn < num_frames - 1:
            next_frame_idx = fn + 1
            # 查找在 fn 后一帧活跃的其他 track
            next_tracks = [t for t in all_tracks if next_frame_idx in t.frames]
            
            # 逻辑 C: 检查是否存在位置重合但类别不同的 track (类别变化)
            found_class_flip_end = False
            found_class_flip_end = match_box(next_tracks, track, found_class_flip_end, 
                                             check_prev=False,
                                             another_frame_idx=next_frame_idx)
            
            # 逻辑 D: 如果后面没有承接的 track，视为纯粹的“目标消失”
            if not found_class_flip_end:
                # 规则：next track 无该目标 -> next track所有帧 +1, fn +1
                track.scores += 1              
                for nt in next_tracks:
                    nt.score_first += 1
    return all_tracks

def aggre_score_in_obj_tracks(all_tracks):
    
    frames_to_aggre = {}
    for track in all_tracks:
        if not track.has_object():
            continue
        
        for f in track.frames:
            curr_score = track.get_frame_score(f)
            if not f in frames_to_aggre:
                frames_to_aggre[f] = curr_score
            else:
                frames_to_aggre[f] += curr_score
    
    for track in all_tracks:
        if not track.has_object():
            continue
        
        for f in track.frames:
            curr_score = track.get_frame_score(f)
            track.aggre_scores[f] = curr_score + frames_to_aggre[f]
    return all_tracks
        
def get_all_sample_score(clip_model, preprocess, text, all_tracks, all_frames):
    '''
    得到每个frame的分数，交叉track的分数叠加
    '''
    
    imgs_entropy = []
    for img in all_frames:
        pil_img = Image.fromarray(img.astype(np.uint8))
        image = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(image, text)
            clip_score_v = logits_per_image
            clip_prob = torch.softmax(clip_score_v, dim=1)
            clip_entropy = torch.sum(-torch.log(clip_prob + 1e-8) * clip_prob, dim=1).cpu().numpy()
            imgs_entropy.append(clip_entropy)
            
    for track in all_tracks:
        if not track.has_object():
            # 对无目标样本，打分选择熵最大的
            
            for f in track.frames:
                clip_entropy = imgs_entropy[f]
                curr_score = track.get_frame_score(f)
                track.aggre_scores[f] = curr_score + float(clip_entropy)
            
        else:
            # 有目标样本，加上预测分数；
            for f in track.frames:
                score = track.history[f].scores.cpu().numpy()
                curr_score = track.get_frame_score(f)
                track.aggre_scores[f] += curr_score + 1- float(score)

    return all_tracks

def get_specify_samples(all_tracks):
    
    # 先在所有frames中采集最多3张图；
    samples = []
    samples_score= {}
    for track in all_tracks:
        for f in track.frames:
            if f not in samples_score: samples_score[f] = track.aggre_scores[f]
            else: samples_score[f] = max(track.aggre_scores[f], samples_score[f])
    
    sorted_frames = [k for k,v in sorted(samples_score.items(), key=lambda x:x[1], reverse=True)]
    samples.extend(sorted_frames[:3])
    
    # 如果存在track没有采集到，则每个track增加一张
    for track in all_tracks:
        sample_in_track = False
        for f in samples:
            if f in track.frames:
                sample_in_track = True
                break
        
        if not sample_in_track:
            sorted_frames_ = [k for k,v in sorted(track.aggre_scores.items(), key=lambda x:x[1], reverse=True)]
            samples.extend(sorted_frames_[:1])
    return samples

iou_threshold=0.4

# 加载数据
data_pth = "outputs_asample/imgs/test5_env1/rgb_all_data"
sampler = SampleLoader(data_pth, glbstep=True)
inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  

# 遍历每一glbstep的数据
# glb_frames = {}
# glb_frame = []

# for env, episode, glbstep, step in zip(inputs[0], inputs[1], inputs[2], inputs[3]):
    
#     if env not in glb_frames:
#         glb_frames[env] = {}
#     if episode not in glb_frames[env]:
#         glb_frames[env][episode] = {}
#     if glbstep not in glb_frames[env][episode]:
#         glb_frames[env][episode][glbstep] = [step]
#     else:
#         glb_frames[env][episode][glbstep].append(step)
# # print(glb_frames)
# with open("./asample_straight_indices_2.pkl", "wb") as f:
#     pickle.dump(glb_frames, f)


glb_frames_tracks = {}
glb_frames_sampled = []
mod = ["bbsgt" , "bbspred", "rgb",]     #  "depth", "position", "semantic", ]

with open("./asample_straight_indices_2.pkl", "rb") as f:
    glb_frames_indices = pickle.load(f)
    
device = "cuda:3" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
text_str = [key for key in target_coco_categories.keys()]
text = clip.tokenize([f"a photo contains a {i}" for i in text_str]).to(device)


#组合每一track并进行采样
for env, env_data in glb_frames_indices.items():
    
    if env not in glb_frames_tracks:
        glb_frames_tracks[env] = {}
        
    for episode, epi_data in env_data.items():
        
        if episode not in glb_frames_tracks[env]:
            glb_frames_tracks[env][episode] = {}
            
        for glbstep, frames in epi_data.items():
            
            
            glb_datas = []
            for f in sorted(frames):
                sample_data = sampler.get_sample_multimodality(
                    env, episode, f, mod, glbstep)
                glb_datas.append(sample_data)
                
            # 对每一glbstep的数据进行分组；同一位置且统一类别为同一组
            
            frame_detections = [data['bbspred'] for data in glb_datas]
            frame_rgbs = [data['rgb'].data for data in glb_datas]
            
            s = time.time()
            groups = group_by_object_and_score(frame_detections)
            print(f"1-{time.time() - s} ")
            
            s = time.time()
            groups = score_tracks(groups, frame_detections)
            print(f"2-{time.time() - s} ")
            
            s = time.time()
            groups = aggre_score_in_obj_tracks(groups)
            print(f"3-{time.time() - s} ")
            
            s = time.time()
            # 从每个track中提取最高分的图像；
            groups = get_all_sample_score(clip_model, preprocess, text, groups, frame_rgbs)
            print(f"4-{time.time() - s} ")
            
            s = time.time()
            
            glb_frames_tracks[env][episode][glbstep] = groups   # 存储打好分的group
            
            sampled_frames = get_specify_samples(groups)
            
            for sf in sampled_frames:
                glb_frames_sampled.append([env, episode, glbstep, sf])
            
            print(f"add tracks of env{env} epi{episode} glbstep{glbstep}")
        # 初始化

    
    
with open("./asample_straight_tracks_2.pkl", "wb") as f:
    pickle.dump(glb_frames_tracks, f)
    
with open("./asample_straight_sampled_2.pkl", "wb") as f:
    pickle.dump(glb_frames_sampled, f)
print(glb_frames_sampled)

