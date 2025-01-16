from detectron2.structures.instances import Instances
from typing import List

def get_objects_ids(batch, predictions: List[Instances], overlap_thr=0.5):
    """
    batch: dictionary containing field "instances" for ground-truth instances and "episode" for current episode
    predictions: Instances object containing model prediction
    """

    gt_labels = []

    # compute y_gt with unique ids starting from 0

    for idx in range(len(predictions)):

        pred = predictions[idx]
        gt = batch[idx]['instances'].to('cpu')

        # Get instance ids for objects
        ids = _get_objects_unique_ids_impl(
            pred,
            gt,
            device='cpu',
            thr=overlap_thr,
            episode=batch[idx]['episode'],
        )

        gt_labels.append(ids)

    return gt_labels


def _get_objects_unique_ids_impl(predictions, gt, device="cuda", thr=0.3, episode=-1):
    """
    Given N predictions and M ground-truth, returns list of length N of unique instance ids given for each prediction. If no ground-truth / prediction matching occurs, id is -1
    """
    if not hasattr(get_objects_ids, "current_unique_id"):
        get_objects_ids.current_unique_id = 5000000
    results = []
    # Use masks IOU for matching gt and preds

    dummy_ids = []
    for _ in range(len(predictions)):
        dummy_ids.append(
            {"id_object": get_objects_ids.current_unique_id, "episode": episode}
        )
        get_objects_ids.current_unique_id += 1
    return dummy_ids