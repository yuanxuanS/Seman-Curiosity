from PIL import Image
import torch
from src.policy_rl.agents.utils.detect_utils import box_iou_calc

iou_threshold=0.4

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
    - 图像中连续无目标
    - 某个物体（类别不变）连续出现
    
    """
    num_frames = len(sequence_detections)
    all_tracks = []
    active_tracks = []

    # --- 阶段 1: 轨迹关联 (把框归类到物体) ---
    for t in range(num_frames):
        curr_objs = sequence_detections[t]
        matched_indices = set()
        matched_curr = False
        
        # 将活跃轨迹匹配当前帧的框
        for track in active_tracks:
            last_f = track.frames[-1]
            last_data = track.history[last_f]
            
            if last_data is None:       # 该track无目标
                if len(curr_objs) == 0:
                    track.history[t] = None
                    track.frames.append(t)
                    matched_curr = True
                else:       # 但新出现目标
                    track.is_active = False
                break       # 如果活跃轨迹有无目标的，则其他
            else:
                if len(curr_objs) == 0: # 当前帧无任何目标，均创建新track
                    track.is_active = False
                    continue
                
                # 当前帧有目标，track和每个目标，找最匹配
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
                    track.is_active = False # 该track未匹配到任一目标，认为轨迹结束
        
        if len(curr_objs) == 0:
            if matched_curr == False:       # 建立无目标帧track
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
    '''
    给每个track打分
    检查track第一帧的前一帧, track的最后一帧的后一帧：
        - 类别变化： 相邻帧均+1
        - 无对应目标： 后一track为目标消失，后一track的score+1; 本track的score+1
        - 目标消失： 前组中选最高； 后组的第一帧入选

    '''
    num_frames = len(sequence_detections)
    
    def match_box(tracks, tgt_track,found_bool, check_prev=False, another_frame_idx=None):
        flip_track = None
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
                    flip_track = pt
                    break # 找到一个匹配的翻转即可
        return found_bool, flip_track
        
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
            found_class_flip, flip_track = match_box(prev_tracks, track, found_class_flip, 
                                         check_prev=True,
                                         another_frame_idx=prev_frame_idx)
                    
            # 逻辑 B: 如果没有类别变化的 track，很有可能未检出
            if not found_class_flip:
                # 两个track均+1
                for pt in prev_tracks:
                    pt.scores += 1
                track.scores += 1
            else:   
                # 找到对应类别变化，上一track最后一帧的score+1， 本track第一帧+1
                flip_track.score_last += 1
                track.score_first += 1
        # --- 2. 检查结尾 (目标消失逻辑) ---
        if fn < num_frames - 1:
            next_frame_idx = fn + 1
            # 查找在 fn 后一帧活跃的其他 track
            next_tracks = [t for t in all_tracks if next_frame_idx in t.frames]
            
            # 逻辑 C: 检查是否存在位置重合但类别不同的 track (类别变化)
            found_class_flip_end = False
            found_class_flip_end, flip_track = match_box(next_tracks, track, found_class_flip_end, 
                                             check_prev=False,
                                             another_frame_idx=next_frame_idx)
            
            # 逻辑 D: 如果后面没有承接的 track，“目标消失” or 未检出
            if not found_class_flip_end:
                # 规则：后组第一帧+1， 前组均+1
                track.scores += 1              
                for nt in next_tracks:
                    nt.score_first += 1
            else:
                flip_track.score_first += 1
                track.score_last += 1
    return all_tracks

def aggre_score_in_obj_tracks(all_tracks, mode='frame'):
    '''
    按照目标track, 给每张图叠加分数
    '''
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
    
    if mode == "frame":
        return frames_to_aggre
    # 将图像分数加回track中，用于后续从track中选样本
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