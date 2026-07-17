from PIL import Image
import numpy as np
import torch

from src.policy_rl.agents.utils.detect_utils import box_iou_calc


STC_IOU_THRESHOLD = 0.5
LEGACY_STC_IOU_THRESHOLD = 0.4
DEFAULT_SAMPLE_BUDGET = 3
DEFAULT_CLIP_TARGET_THRESHOLD = 0.7


class ObjectTrack:
    """Detection trajectory tau_p for one object in a meta-sequence."""

    def __init__(self, start_frame, detection):
        self.history = {start_frame: detection}
        self.frames = [start_frame]
        self.start_frame = start_frame
        self.end_frame = start_frame
        self.bd_scores = {}
        self.cls_scores = {}
        self.unc_scores = {}
        self.scores = 0.0
        self.score_first = 0.0
        self.score_last = 0.0
        self.aggre_scores = {}
        self.best_frames = {}
        self.is_active = True

    def add_detection(self, frame_idx, detection):
        self.history[frame_idx] = detection
        self.frames.append(frame_idx)
        self.end_frame = frame_idx

    def has_object(self):
        return bool(self.history) and next(iter(self.history.values())) is not None

    def get_frame_score(self, frame_idx):
        assert frame_idx in self.frames, f"frame {frame_idx} not in track"
        if frame_idx == min(self.frames):
            return self.scores + self.score_first
        if frame_idx == max(self.frames):
            return self.scores + self.score_last
        return self.scores


class STCSelectionResult(list):
    """Object tracks plus frame-level STC scores."""

    def __init__(self, tracks=None, num_frames=0):
        super().__init__(tracks or [])
        self.num_frames = num_frames
        self.frame_scores = {frame_idx: 0.0 for frame_idx in range(num_frames)}
        self.clip_fallback = False
        self.clip_candidate_frames = None


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _detection_box(detection):
    return _to_numpy(detection.pred_boxes.tensor)


def _detection_class(detection):
    return int(_to_numpy(detection.pred_classes).reshape(-1)[0])


def _detection_score(detection):
    return float(_to_numpy(detection.scores).reshape(-1)[0])


def _detection_iou(detection_a, detection_b):
    return float(box_iou_calc(_detection_box(detection_a), _detection_box(detection_b)).max())


def _frame_has_spatial_match(frame_detections, target_detection, iou_threshold):
    for detection_idx in range(len(frame_detections)):
        try:
            iou = _detection_iou(frame_detections[detection_idx], target_detection)
        except AttributeError:
            return False
        if iou > iou_threshold:
            return True
    return False


def group_by_object_and_score(
    sequence_detections, iou_threshold=LEGACY_STC_IOU_THRESHOLD
):
    """
    Legacy STC step 1: group frames into object/no-object tracks.

    This is the algorithm used before the STC rewrite. It links detections only
    when IoU passes the threshold and the predicted class is unchanged, and it
    also creates tracks for consecutive frames with no detections.
    """
    num_frames = len(sequence_detections)
    all_tracks = []
    active_tracks = []

    for frame_idx in range(num_frames):
        curr_objs = sequence_detections[frame_idx]
        matched_indices = set()
        matched_curr = False

        for track in active_tracks:
            last_frame = track.frames[-1]
            last_data = track.history[last_frame]

            if last_data is None:
                if len(curr_objs) == 0:
                    track.history[frame_idx] = None
                    track.frames.append(frame_idx)
                    track.end_frame = frame_idx
                    matched_curr = True
                else:
                    track.is_active = False
                break

            if len(curr_objs) == 0:
                track.is_active = False
                continue

            best_iou = 0.0
            best_idx = -1
            for detection_idx in range(len(curr_objs)):
                if detection_idx in matched_indices:
                    continue

                curr_box = _detection_box(curr_objs[detection_idx])
                last_box = _detection_box(last_data)
                iou = float(box_iou_calc(last_box, curr_box).max())
                if (
                    iou > iou_threshold
                    and iou > best_iou
                    and _detection_class(curr_objs[detection_idx])
                    == _detection_class(last_data)
                ):
                    best_iou = iou
                    best_idx = detection_idx

            if best_idx != -1:
                track.add_detection(frame_idx, curr_objs[best_idx])
                matched_indices.add(best_idx)
            else:
                track.is_active = False

        if len(curr_objs) == 0:
            if not matched_curr:
                new_track = ObjectTrack(frame_idx, None)
                all_tracks.append(new_track)
                active_tracks.append(new_track)
        else:
            for detection_idx in range(len(curr_objs)):
                if detection_idx in matched_indices:
                    continue
                new_track = ObjectTrack(frame_idx, curr_objs[detection_idx])
                all_tracks.append(new_track)
                active_tracks.append(new_track)

        active_tracks = [track for track in active_tracks if track.is_active]

    return all_tracks


def score_tracks(
    all_tracks, sequence_detections, iou_threshold=LEGACY_STC_IOU_THRESHOLD
):
    """
    Legacy STC step 2: score track starts/ends and class flips.

    Boundary gaps add value to adjacent tracks. If adjacent tracks overlap in
    space but change class, only the boundary frames receive class-flip value.
    """
    num_frames = len(sequence_detections)

    def match_box(tracks, tgt_track, check_prev=False, another_frame_idx=None):
        curr_frame = min(tgt_track.frames)
        curr_label = _detection_class(tgt_track.history[curr_frame])
        curr_box = _detection_box(tgt_track.history[curr_frame])

        for other_track in tracks:
            if not other_track.has_object():
                continue
            other_det = other_track.history[another_frame_idx]
            other_box = _detection_box(other_det)
            if float(box_iou_calc(other_box, curr_box).max()) <= iou_threshold:
                continue
            if _detection_class(other_det) == curr_label:
                continue

            if check_prev:
                other_track.score_last += 1
                tgt_track.score_first += 1
            else:
                other_track.score_first += 1
                tgt_track.score_last += 1
            return True, other_track
        return False, None

    for track in all_tracks:
        if not track.has_object():
            continue

        first_frame = min(track.frames)
        last_frame = max(track.frames)

        if first_frame > 0:
            prev_frame = first_frame - 1
            prev_tracks = [t for t in all_tracks if prev_frame in t.frames]
            found_class_flip, flip_track = match_box(
                prev_tracks, track, check_prev=True, another_frame_idx=prev_frame
            )

            if not found_class_flip:
                for prev_track in prev_tracks:
                    prev_track.scores += 1
                track.scores += 1
            else:
                flip_track.score_last += 1
                track.score_first += 1

        if last_frame < num_frames - 1:
            next_frame = last_frame + 1
            next_tracks = [t for t in all_tracks if next_frame in t.frames]
            found_class_flip, flip_track = match_box(
                next_tracks, track, check_prev=False, another_frame_idx=next_frame
            )

            if not found_class_flip:
                track.scores += 1
                for next_track in next_tracks:
                    next_track.score_first += 1
            else:
                flip_track.score_first += 1
                track.score_last += 1

    return all_tracks


def aggre_score_in_obj_tracks(all_tracks, mode="frame"):
    """Legacy STC step 3: aggregate object-track scores to frames."""
    frames_to_aggre = {}
    for track in all_tracks:
        if not track.has_object():
            continue

        for frame_idx in track.frames:
            curr_score = track.get_frame_score(frame_idx)
            frames_to_aggre[frame_idx] = frames_to_aggre.get(frame_idx, 0.0) + curr_score

    if mode == "frame":
        return frames_to_aggre

    for track in all_tracks:
        if not track.has_object():
            continue

        for frame_idx in track.frames:
            curr_score = track.get_frame_score(frame_idx)
            track.aggre_scores[frame_idx] = curr_score + frames_to_aggre[frame_idx]
    return all_tracks


def get_all_sample_score(clip_model, preprocess, text, all_tracks, all_frames, device):
    """Legacy STC step 4: add CLIP entropy for empty tracks and 1-score for objects."""
    imgs_entropy = []
    if clip_model is not None and preprocess is not None and text is not None:
        for frame in all_frames:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            image = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits_per_image, _ = clip_model(image, text)
                clip_prob = torch.softmax(logits_per_image, dim=1)
                clip_entropy = torch.sum(
                    -torch.log(clip_prob + 1e-8) * clip_prob, dim=1
                )
                imgs_entropy.append(
                    float(clip_entropy.detach().cpu().numpy().reshape(-1)[0])
                )
    else:
        imgs_entropy = [0.0 for _ in all_frames]

    for track in all_tracks:
        if not track.has_object():
            for frame_idx in track.frames:
                curr_score = track.get_frame_score(frame_idx)
                track.aggre_scores[frame_idx] = curr_score + imgs_entropy[frame_idx]
        else:
            for frame_idx in track.frames:
                score = _detection_score(track.history[frame_idx])
                curr_score = track.get_frame_score(frame_idx)
                track.aggre_scores[frame_idx] = (
                    track.aggre_scores.get(frame_idx, 0.0) + curr_score + 1.0 - score
                )

    return all_tracks


def get_specify_samples(all_tracks, budget=DEFAULT_SAMPLE_BUDGET):
    """Legacy STC step 5: pick top frames, then cover tracks not represented."""
    samples = []
    samples_score = {}
    for track in all_tracks:
        for frame_idx in track.frames:
            if frame_idx not in track.aggre_scores:
                continue
            samples_score[frame_idx] = max(
                track.aggre_scores[frame_idx], samples_score.get(frame_idx, float("-inf"))
            )

    sorted_frames = [
        frame_idx
        for frame_idx, _ in sorted(
            samples_score.items(), key=lambda item: item[1], reverse=True
        )
    ]
    samples.extend(sorted_frames[:budget])

    for track in all_tracks:
        if any(frame_idx in track.frames for frame_idx in samples):
            continue
        sorted_track_frames = [
            frame_idx
            for frame_idx, _ in sorted(
                track.aggre_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]
        samples.extend(sorted_track_frames[:1])
    return samples


def _ensure_stc_result(tracks, num_frames):
    if isinstance(tracks, STCSelectionResult):
        tracks.num_frames = num_frames
        return tracks
    return STCSelectionResult(tracks, num_frames=num_frames)


def extract_object_tracks(
    sequence_detections,
    iou_threshold=STC_IOU_THRESHOLD,
    group_empty_tracks=False,
):
    """
    Step 1: extract object trajectories from adjacent-frame spatial consistency.

    Detections in frame k and k+1 are assigned to the same trajectory when
    IoU(b_k, b_{k+1}) is larger than q. Class labels are deliberately ignored
    here so that class changes can be scored later. When ``group_empty_tracks``
    is enabled, consecutive frames without detections form one empty trajectory.
    """
    num_frames = len(sequence_detections)
    tracks = STCSelectionResult(num_frames=num_frames)
    active_tracks = []
    empty_track = None

    for frame_idx, frame_detections in enumerate(sequence_detections):
        matched_detection_indices = set()

        if group_empty_tracks and len(frame_detections) == 0:
            if empty_track is None:
                empty_track = ObjectTrack(frame_idx, None)
                tracks.append(empty_track)
            else:
                empty_track.add_detection(frame_idx, None)
        else:
            empty_track = None

        for track in active_tracks:
            last_frame = track.frames[-1]
            if last_frame != frame_idx - 1:
                track.is_active = False
                continue

            best_iou = 0.0
            best_detection_idx = -1
            for detection_idx in range(len(frame_detections)):
                if detection_idx in matched_detection_indices:
                    continue
                iou = _detection_iou(
                    track.history[last_frame], frame_detections[detection_idx]
                )
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_detection_idx = detection_idx

            if best_detection_idx == -1:
                track.is_active = False
            else:
                track.add_detection(frame_idx, frame_detections[best_detection_idx])
                matched_detection_indices.add(best_detection_idx)

        active_tracks = [track for track in active_tracks if track.is_active]

        for detection_idx in range(len(frame_detections)):
            if detection_idx in matched_detection_indices:
                continue
            track = ObjectTrack(frame_idx, frame_detections[detection_idx])
            tracks.append(track)
            active_tracks.append(track)

    return tracks


def score_boundary_missing_detections(
    tracks,
    num_frames,
    sequence_detections=None,
    iou_threshold=STC_IOU_THRESHOLD,
):
    """
    Step 2: score potential missing detections around trajectory boundaries.

    V_bd(i_k, tau_p) = 1(k < a_p) + 1(k = b_p + 1), where a_p is the first
    trajectory frame and b_p is the last trajectory frame.

    For non-adjacent predecessor frames, if a detection already exists at the
    same spatial position as the trajectory start, the frame is not counted as a
    missing detection.
    """
    tracks = _ensure_stc_result(tracks, num_frames)
    tracks.frame_scores = {frame_idx: 0.0 for frame_idx in range(num_frames)}

    for track in tracks:
        track.start_frame = min(track.frames)
        track.end_frame = max(track.frames)
        track.bd_scores = {}
        track.cls_scores = {}
        track.unc_scores = {}

        if not track.has_object():
            continue

        start_detection = track.history[track.start_frame]
        adjacent_prev_frame = track.start_frame - 1
        for frame_idx in range(track.start_frame):  # 前序帧值，漏检+1
            if (
                sequence_detections is not None
                and frame_idx != adjacent_prev_frame
                and _frame_has_spatial_match(
                    sequence_detections[frame_idx], start_detection, iou_threshold
                )
            ):
                continue
            track.bd_scores[frame_idx] = track.bd_scores.get(frame_idx, 0.0) + 1.0
            tracks.frame_scores[frame_idx] += 1.0 #

        next_frame = track.end_frame + 1
        if next_frame < num_frames:  # 后序的一帧+1
            track.bd_scores[next_frame] = track.bd_scores.get(next_frame, 0.0) + 1.0
            tracks.frame_scores[next_frame] += 1.0

    return tracks


def score_class_changes_in_tracks(tracks):
    """
    Step 3: score class changes between adjacent detections in each trajectory.

    If c_{k,p} != c_{k+1,p}, both frames receive one value point.
    """
    for track in tracks:
        if not track.has_object():
            continue
        for prev_frame, curr_frame in zip(track.frames[:-1], track.frames[1:]):
            if curr_frame != prev_frame + 1:
                continue
            if _detection_class(track.history[prev_frame]) == _detection_class(
                track.history[curr_frame]
            ):
                continue

            track.cls_scores[prev_frame] = track.cls_scores.get(prev_frame, 0.0) + 1.0
            track.cls_scores[curr_frame] = track.cls_scores.get(curr_frame, 0.0) + 1.0
            tracks.frame_scores[prev_frame] += 1.0
            tracks.frame_scores[curr_frame] += 1.0

    return tracks


def score_prediction_uncertainty(tracks):
    """
    Step 4: score detection uncertainty inside trajectories.

    V_unc(i_k, tau_p) = 1 - s_{k,p}.
    """
    for track in tracks:
        if not track.has_object():
            continue
        for frame_idx in track.frames:
            unc_score = 1.0 - _detection_score(track.history[frame_idx])
            track.unc_scores[frame_idx] = unc_score
            tracks.frame_scores[frame_idx] += unc_score

    return tracks


def compute_clip_entropy_scores(
    clip_model,
    preprocess,
    text,
    frames,
    device,
    target_threshold=DEFAULT_CLIP_TARGET_THRESHOLD,
):
    """Fallback value for frames whose max CLIP class probability is confident."""
    frame_scores = {}
    candidate_frames = set()
    for frame_idx, frame in enumerate(frames):
        pil_img = Image.fromarray(frame.astype(np.uint8))
        image = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, _ = clip_model(image, text)
            class_probs = torch.softmax(logits_per_image, dim=1)
            max_prob = torch.max(class_probs, dim=1).values
            entropy = torch.sum(
                -torch.log(class_probs + 1e-8) * class_probs, dim=1
            )
        if float(max_prob.detach().cpu().numpy().reshape(-1)[0]) > target_threshold:
            frame_scores[frame_idx] = float(entropy.detach().cpu().numpy().reshape(-1)[0])
            candidate_frames.add(frame_idx)
    return frame_scores, candidate_frames


def add_uncertainty_or_clip_fallback(
    clip_model,
    preprocess,
    text,
    tracks,
    frames,
    device,
    clip_target_threshold=DEFAULT_CLIP_TARGET_THRESHOLD,
):
    """Apply V_unc, or CLIP entropy when there are no object tracks."""
    tracks = _ensure_stc_result(tracks, len(frames))

    if not any(track.has_object() for track in tracks):
        tracks.frame_scores, tracks.clip_candidate_frames = compute_clip_entropy_scores(
            clip_model,
            preprocess,
            text,
            frames,
            device,
            target_threshold=clip_target_threshold,
        )
        tracks.clip_fallback = True
        return tracks

    tracks = score_prediction_uncertainty(tracks)
    tracks.clip_fallback = False
    tracks.clip_candidate_frames = None
    return tracks


def select_top_value_samples(tracks, budget=DEFAULT_SAMPLE_BUDGET):
    """Step 5: select Top-B frames, then cover every unrepresented track."""
    frame_scores = dict(getattr(tracks, "frame_scores", {}))
    if getattr(tracks, "clip_fallback", False):
        candidate_frames = getattr(tracks, "clip_candidate_frames", None)
        if candidate_frames is not None:
            frame_scores = {
                frame_idx: frame_scores[frame_idx]
                for frame_idx in candidate_frames
                if frame_idx in frame_scores
            }

    num_frames = getattr(tracks, "num_frames", len(frame_scores))
    if not getattr(tracks, "clip_fallback", False):
        for frame_idx in range(num_frames):
            frame_scores.setdefault(frame_idx, 0.0)

    sorted_frames = [
        frame_idx
        for frame_idx, _ in sorted(
            frame_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True
        )
    ]
    samples = sorted_frames[: min(budget, len(sorted_frames))]
    selected_frames = set(samples)

    # Top-B selection can leave an object trajectory entirely unrepresented.
    # Add the highest-value frame from each such track so every trajectory
    # contributes at least one frame to the sampled dataset.
    for track in tracks:
        if selected_frames.intersection(track.frames):
            continue

        best_frame = max(
            track.frames,
            key=lambda frame_idx: (frame_scores.get(frame_idx, 0.0), -frame_idx),
        )
        if best_frame not in selected_frames:
            samples.append(best_frame)
            selected_frames.add(best_frame)

    return samples


def stc_select_samples(
    sequence_detections,
    frames,
    budget=DEFAULT_SAMPLE_BUDGET,
    clip_model=None,
    preprocess=None,
    text=None,
    device=None,
    iou_threshold=STC_IOU_THRESHOLD,
    clip_target_threshold=DEFAULT_CLIP_TARGET_THRESHOLD,
    algorithm="rewrite",
    group_empty_tracks=False,
):
    """Run the full STC-selection algorithm on one meta-sequence."""
    if algorithm == "legacy":
        tracks = group_by_object_and_score(
            sequence_detections, iou_threshold=LEGACY_STC_IOU_THRESHOLD
        )
        tracks = score_tracks(
            tracks, sequence_detections, iou_threshold=LEGACY_STC_IOU_THRESHOLD
        )
        tracks = aggre_score_in_obj_tracks(tracks, mode="track")
        tracks = get_all_sample_score(
            clip_model, preprocess, text, tracks, frames, device
        )
        samples = get_specify_samples(tracks, budget=budget)
        return tracks, samples

    if algorithm != "rewrite":
        raise ValueError(f"Unsupported STC algorithm: {algorithm}")

    tracks = extract_object_tracks(
        sequence_detections,
        iou_threshold=iou_threshold,
        group_empty_tracks=group_empty_tracks,
    )
    tracks = score_boundary_missing_detections(
        tracks,
        len(sequence_detections),
        sequence_detections=sequence_detections,
        iou_threshold=iou_threshold,
    )
    tracks = score_class_changes_in_tracks(tracks)
    tracks = add_uncertainty_or_clip_fallback(
        clip_model,
        preprocess,
        text,
        tracks,
        frames,
        device,
        clip_target_threshold=clip_target_threshold,
    )
    samples = select_top_value_samples(tracks, budget=budget)
    return tracks, samples
