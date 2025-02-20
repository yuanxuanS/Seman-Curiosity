from detectron2.structures import Instances, Boxes
from torchmetrics.detection.map import MAP
import pytorch_lightning as pl
import torch
import itertools as it
from torch.nn import functional as F
import gc
import tqdm
import time
import logging
import cv2
import copy
from src.finetune.utils.matching import get_objects_ids
from src.finetune.utils import projection_utils as pu
from src.finetune.sensors_data import BBSense
from src.policy_rl.agents.utils.semantic_prediction import ImageSegmentation
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

log = logging.getLogger(__name__)

def map_to_original_cls(instance, cls):
    '''
    cls: {0: 57, 1:58, ...}
    '''
    new_instance = copy.deepcopy(instance)

    for i in range(len(instance.pred_classes)):
        new_idx = cls[int(instance.pred_classes[i])]
        new_instance.pred_classes[i] = new_idx
    new_instance.pred_classes = torch.tensor(new_instance.pred_classes)
    return new_instance

class ConsensusLabeler(pl.LightningModule):
    def __init__(self, 
                 model=None, 
                 thr=0.7, 
                 *args, 
                 **kwargs):
        super().__init__()
        
        self.thr = thr
        self.reinit(model)
        
        
    def reinit(self, model=None):
        self.update_model(model)
        self.test_map_metric = MAP(class_metrics=True)
        
    def update_model(self, model):
        self.model = model
        self.model.model.roi_heads.box_predictor.test_score_thresh = (
            self.thr
        )
        self.model.eval()
        
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.model.eval()
        instances = self(batch)

        return instances
    
    def forward(self, batch):
        # self.model.to(f"cuda:{self.device_id}")
        self.model.eval()
        preds = [x['instances'].to('cpu') for x in self.model(batch)[0]]
        ids = get_objects_ids(batch, preds)

        return preds, ids
    
    def get_pseudo_labels(self, *args, **kwargs):
        pass
    
class VanillaConsensusLabeler(ConsensusLabeler):
    '''
        直接用预测的类别作为 目标类别
    '''
    def __init__(self, model=None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        
    def get_pseudo_labels(self, model_outs, *args, **kwargs):   # TODO
        """
        Returns predictions as pseudo ground-truth
        """
        result = []
        for out in model_outs:
            for pred, infos in zip(out[0], out[1]):

                mask = pred.scores > 0.0001
                pred = pred[mask]
                target = Instances(len(pred))
                target.gt_classes = pred.pred_classes
                target.gt_boxes = pred.pred_boxes
                if len(pred) > 0:
                    target.gt_logits = pred.gt_logits
                else:
                    target.gt_logits = torch.Tensor()

                target.scores = pred.scores
                target.gt_masks = pred.pred_masks

                target.infos = [info for idx, info in enumerate(infos) if mask[idx]]
                result.append(target)
        return result


class SemanticConsensusLabeler(ConsensusLabeler):
    def __init__(
        self, model=None, solution="ours", *args, **kwargs
    ):
        super().__init__(model=model, *args, **kwargs)
        self.solution = solution
        
        self.img_pth = ""

        self.args = 
    
        self.obns_model = ImageSegmentation(self.args)
        
    def reinit(self, model):
        super().reinit(model)
        self.global_pcds = {}

    def get_obns_prediction(self, 
                            img, 
                            idx: int, 
                            depth,
                            rcnn_instance: Instances, save=False):
        args = self.args
        image_list = []
        # img = img[:, :, ::-1]
        image_list.append(img)
        obns_instance, vis_output = self.obns_model.get_predictions(
            image_list, visualize=True, specify_cls=True
        )
        
        # find potential one
            
        device = None
        
        obns_instance = obns_instance[0]['instances'].to("cpu")
        
        cnt = 0
        
        width, height = obns_instance.image_size
        pot_mp = np.zeros((height, width, 1))
        v = Visualizer(pot_mp)
        
        # pot_surro_mask = np.zeros((height, width))
        
        objectness_boxes = obns_instance.pred_boxes
        
        # no objectness prediction
        if not len(objectness_boxes):  
            # pot_mp = cv2.resize(pot_mp, (self.args.frame_height, self.args.frame_width))[..., np.newaxis]   # TODO
            return 
            
        for j in range(len(objectness_boxes)):
            boxes_ = v._convert_boxes(objectness_boxes[j]).reshape(4,) # convert from 1*4 to 4*1
            
            # remove close boxes
            depth_patch = self.get_patch_from_depth(depth, boxes_)
            if depth_patch.max() < 0.1:  # 
                continue
                
            # remove box that detected by maskrcnn as well
            maskrcnn_boxes = rcnn_instance.pred_boxes
            if len(maskrcnn_boxes):
                # recurse every maskrcnn's boxes to filter IoU > thes:
                device = rcnn_instance.pred_boxes.device
                rcnn_instance = rcnn_instance.to("cpu")
                maskrcnn_boxes = rcnn_instance.pred_boxes
                
                obns_ = v._convert_boxes(objectness_boxes[j])
                msk_ = v._convert_boxes(maskrcnn_boxes)     # all maskrcnn box 
                iou = box_iou_calc(obns_, msk_) # 1*num_maskbox
                if (iou > 0.5).any():   # detected by maskrcnn as well, remove it
                    continue
            print("has far object")
            cnt += 1
            if save:
                cv2.imwrite(self.img_pth + "/obns_imgs/img_"+str(idx)+".png", vis_output.get_image())
            # pot_mp = v.draw_patch(box_coord=boxes_, color='white')
            # mask_ = obns_instance.pred_masks[j]
            # pot_surro_mask[mask_.cpu().numpy() >0] = 1.
        
        # resize to 128*128
        # if isinstance(pot_mp, np.ndarray):
        #     pot_mp = pot_mp.squeeze(-1) if len(pot_mp.shape) == 3 else pot_mp
        # else:
        #     pot_mp = pot_mp.get_image()
        
        # pot_mp[pot_mp > 0] = 1.
        # pot_mp= (pot_mp.astype('float32')*depth)[..., np.newaxis]    # multiply depth
        
        # pot_surro_mask = pot_surro_mask[..., np.newaxis]
        # cv2.imwrite(f"/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/t_potential_d.png", pot_mp.transpose(1,2,0))
        
        # if device is not None:
        #     self.seg_instances[0]['instances'] = self.seg_instances[0]['instances'].to(device)
        # return pot_mp, cnt, pot_surro_mask
        
        
        
        
        
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        '''
            构建pcd
        '''
        self.model.eval()
        instances, infos = self(batch)      # infos: instance ids per image

        for b, prediction, info in zip(batch, instances, infos):
            env = b['env']
            episode = b['episode']
            if env in self.global_pcds:
                if episode in self.global_pcds[env]:
                    episode_pcd = self.global_pcds[env][episode]
                else:
                    episode_pcd = pu.SemanticPointCloud(
                    episode=episode, env=env, solution=self.solution
                )
                    self.global_pcds[env][episode] = episode_pcd
            else:
                episode_pcd = pu.SemanticPointCloud(
                    episode=episode, solution=self.solution
                )
                self.global_pcds[env] = {}
                self.global_pcds[env][episode] = episode_pcd

            _pcd = pu.project_semantic_masks_to_3d(
                b['depth'].squeeze(0),      
                b['location'],      
                prediction.to(b['depth'].device),
                info,
                update_logits=False # do not update logits locally, only per episode
            )
            _pcd._episode = episode
            _pcd._env = env

            episode_pcd += _pcd

            if len(_pcd):
                episode_pcd.update_logits(prediction, info)
             #     # episode_pcd.preprocess()
        return instances, infos

    def get_pseudo_labels(self, model_outs, dataloader):
        """
        Returns predictions as pseudo ground-truth
        """

        labels = []

        for env in self.global_pcds.keys():
            for k in self.global_pcds[env].keys():
                self.global_pcds[env][k].preprocess()        # 统一点云的所有标签和类别
        n = 0
        for batch in tqdm.tqdm(dataloader):
            
            for data in batch:
                pcd = self.global_pcds[data['env']][data['episode']]

                # Compute the ray intersections.
                _time = time.time()
                
                (
                    semantic_masks,
                    object_ids,
                    classes,
                    r_logits,
                    _
                ) = pcd._depth_raytracing(data['depth'].squeeze(), data['location'])    # 找到data的同一个object的pcd，查找该标签

                log.info(f"It took {time.time() - _time} for raytracing")
                t = Instances(image_size=data['depth'].squeeze().shape)

                # Get bbs from semantic
                resolved_masks = []
                bounding_boxes = []
                logits = []
                resolved_classes = []
                ids = []

                for mask, object_id, cls_, l in zip(
                    semantic_masks, object_ids, classes, r_logits
                ):
                    bb = cv2.boundingRect(mask.numpy().astype('uint8'))

                    x, y, w, h = bb
                    if w == 0 or h == 0:
                        continue
                    if cls_ >= len(BBSense.CLASSES):        # class的范围？
                        continue  # Background or overflowd class
                    
                    logits.append(l)        # TODO : l/ temperature
                    resolved_class = cls_
                    ids.append(object_id)
                    
                    bounding_boxes.append(
                        torch.tensor([x, y, x + w, y + h]).unsqueeze(0)
                    )
                    resolved_classes.append(resolved_class)
                    resolved_masks.append(mask)

                t.gt_classes = (
                    torch.tensor(resolved_classes)
                    if len(resolved_classes) > 0
                    else torch.Tensor()
                )

                t.gt_masks = (
                    torch.stack(resolved_masks)
                    if len(resolved_masks)
                    else torch.Tensor()
                )
                # TODO uncertainty information per bbox

                t.gt_logits = torch.stack(logits) if len(logits) > 0 else torch.Tensor()

                t.infos = [{'id_object': id.item()} for id in ids]

                t.gt_boxes = (
                    Boxes(torch.cat(bounding_boxes))
                    if len(bounding_boxes)
                    else Boxes(torch.Tensor())
                )
                labels.append(t)
                
                self.save_image(data, t, n)
                n+= 1
                
                # obns prediction
                self.get_obns_prediction(data['rgb'], n, data['depth'], t)
        gc.collect()
        return labels

    def save_image(self, img, instance: Instances, idx: int):
        v = Visualizer(
                img, MetadataCatalog.get('coco_2017_val'))
        cls_id_map = {0: 56, 1:57, 2:58, 3:59, 4:61}
        instances_mapped = map_to_original_cls(instance.to("cpu"), cls_id_map)
        v = v.draw_instance_predictions(instances_mapped)
        img = cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.img_pth + "/rcnn_imgs/img_"+str(idx)+".png")

class LogitsConsensusLabeler(ConsensusLabeler):
    def __init__(self, temperature=1, model=None,*args, **kwargs):
        super().__init__(model,*args, **kwargs)
        self.temperature = temperature
        print(f"temperature is {self.temperature} in pseudolaber")
        
    def get_pseudo_labels(self, model_outs, *args, **kwargs):   # TODO
        """
        Returns predictions as pseudo ground-truth
        """

        predictions = list(it.chain(*[m[0] for m in model_outs]))

        y_ids = list(it.chain(*list(it.chain(*[m[1] for m in model_outs]))))

        max_id = max([m['id_object'] for m in y_ids])

        y_matching = torch.tensor(
            [
                m['id_object'] + m['episode'] * max_id if m['id_object'] > 0 else -1
                for m in y_ids
            ]
        )   # 给每个object 分配一个标记

        preds_logits = torch.cat([pred.gt_logits for pred in predictions])

        match_ids = torch.unique(y_matching)        # object id的集合

        logits_per_instance = {
            m.item(): preds_logits[y_matching == m] for m in match_ids
        }       # 提取object所有的预测logits， {object idx：logits, ...}

        gt_instances = []

        pred_counting = 0

        for idx in range(len(predictions)):

            preds_per_image = predictions[idx]

            if len(preds_per_image) == 0:
                target = Instances(preds_per_image.image_size)
                target.gt_boxes = preds_per_image.pred_boxes
                target.gt_classes = preds_per_image.pred_classes
                target.gt_logits = preds_per_image.gt_logits
                target.gt_masks = preds_per_image.pred_masks
                target.scores = preds_per_image.scores
                target.infos = []

                gt_instances.append(target)
                continue
            resolved_classes = []
            gt_logits = []
            gt_ids = []
            mask = torch.ones(len(preds_per_image), dtype=torch.bool)
            for pred_id in range(len(preds_per_image)):
                p = pred_id + pred_counting

                y = y_matching[p].item()        # 该instance对应的object id

                logits = logits_per_instance[y]
                # breakpoint()
                soft_softmax = F.softmax(logits / self.temperature, -1).mean(0)
                resolved_class = torch.argmax(soft_softmax[:-1])
                score = soft_softmax.max()
                if score < 0.001:
                    mask[pred_id] = False
                else:
                    resolved_classes.append(resolved_class)
                    gt_logits.append(soft_softmax)
                    gt_ids.append({'id_object': y})

            pred_counting += len(preds_per_image)
            target = Instances(preds_per_image[mask].image_size)
            target.gt_boxes = preds_per_image[mask].pred_boxes
            target.gt_classes = (
                torch.tensor(resolved_classes)
                if len(resolved_classes) > 0
                else torch.Tensor()
            )
            target.gt_masks = preds_per_image[mask].pred_masks
            target.gt_logits = (
                torch.stack(gt_logits) if len(gt_logits) > 0 else torch.Tensor()
            )

            target.infos = gt_ids

            gt_instances.append(target)

        return gt_instances