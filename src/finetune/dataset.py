from torch.utils.data import Dataset
import torch
from albumentations.pytorch import ToTensorV2
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks
from .dataset_utils import SampleLoader
from .sensors_data import BBSense
import numpy as np


class BbsgtDataset(Dataset):
    '''
    if not specified,  rgb and bbsgt
    '''
    def __init__(
        self,
        modalities=None,
        data_path=None,
        sampler=None,
        index_mask=None,
        inputs= None,
        transform=None,
        remap_classes=True,
    ):
        super().__init__()
        
        self.modalities = ['rgb', 'bbsgt'] if modalities is None else modalities

        self.data_path = data_path
        self.sampler = SampleLoader(data_path) if sampler is None else sampler
        
        if inputs is None:
            env_list, episode_list, steps_list = self.sampler.get_env_episode_and_steps_dense_list()
            if index_mask:
                env_list = env_list[index_mask]
                episode_list = episode_list[index_mask]
                steps_list = steps_list[index_mask]
                
            self.inputs = np.array([x for x in zip(env_list, episode_list, steps_list)])
        else:
            self.inputs = inputs
            
        self.index = np.arange(len(self.inputs))
        
        self.transform = ToTensorV2() if transform is None else transform
        
        self.remap_classes = remap_classes
        
    def __len__(self):
        return len(self.index)
    
    def _transform_batch(self, x, y):
        
        if isinstance(self.transform, ToTensorV2):
            out = self.transform(image=x)
            x = out['image']
        else:
            transformed = self.transform(
                image=x,
                bboxes=y.gt_boxes.tensor.numpy(),
                class_labels=y.gt_classes,
                masks=[m for m in y.gt_masks.numpy().astype(np.uint8)],
                infos=y.infos,
            )
            
            x = transformed['image']
            
            if len(transformed['class_labels']) > 0:
                y = Instances(
                    image_size=x.shape[1:],
                    gt_boxes=Boxes(transformed['bboxes']),
                    gt_classes=torch.stack(transformed['class_labels']),
                    gt_masks=BitMasks(
                        torch.stack([torch.tensor(x) for x in transformed['masks']])),
                    infos=transformed['infos'],
                )
            else:
                y = Instances(
                    gt_boxes=Boxes(torch.Tensor()),
                    image_size=x.shape[1:],
                    gt_masks=BitMasks(torch.Tensor(size=[0, *x.shape[1:]])),
                    gt_classes=torch.Tensor(),
                    infos=[],
                )
                
        if self.remap_classes:
            y.gt_classes = torch.tensor(
                [BBSense.CLASSES_TO_IDX[x.item()] for x in y.gt_classes]
            )
        return x, y
    
    def __getitem__(self, idx):
        env, episode, step = self.inputs[self.index[idx]]
        
        camera_id = 0
        data = self.sampler.get_sample_multimodality(
            env, episode, step, self.modalities
        )
        
        x = data['rgb'].data
        y = data['bbsgt'].get_bbs_as_gt()       # instance，gt前缀
        x, y = self._transform_batch(x, y)      # TODO: 不需要remap class吗
        
        size = x.shape[1:]
        return {        # TODO; 需要这么多吗
            'env': env,
            'episode': episode,
            'step': step,
            'image': x,
            'image_id': idx,
            'instances': y,
            'width': size[1],
            'height': size[0],
        }

    def get_coco_item_dict(self, idx):
        ind = self.index[idx].item()
        env, episode, step = self.inputs[ind]

        camera_id = 0
        data = self.sampler.get_sample_multimodality(
            env, episode, step, ['bbsgt']
        )
        
        gt = data['bbsgt']
        file_name = gt.frame.sense_info.get_path()
        y = data['bbsgt'].get_bbs_as_gt()
        class_labels = y.gt_classes
        
        if self.remap_classes:
            class_labels = torch.tensor(
                [BBSense.CLASSES_TO_IDX[x.item()] for x in class_labels]
            )
            
        annotations = [
            {
                'bbox': y[id_instance].gt_boxes.tensor[0].tolist(),
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_labels[id_instance],
                # 'segmentation': y[id_instance].gt_masks,
                'iscrowd': 0,
            }
            for id_instance in range(len(y))
        ]
        
        instance_dict = {
            'file_name': file_name,
            'image_id': ind,
            'height': y.image_size[0],
            'width': y.image_size[1],
            'annotations': annotations,
            'env':env,
            'episode': episode,
            'step': step
        }

        return instance_dict
    
class FullDataset(BbsgtDataset):
    '''
        rgb, bbsgt, depth, pose
    '''
    def __init__(self, data_path, *args, **kwargs):
        super().__init__(
            modalities=["rgb", "depth", "position", "bbsgt"], data_path=data_path, *args, **kwargs
        )
        
    def __getitem__(self, idx):
        env, episode, step = self.inputs[self.index[idx]]

        data = self.sampler.get_sample_multimodality(
            env, episode, step, self.modalities
        )

        x = data['rgb'].data
        depth = data['depth'].data
        location = data['position'].get_T()
        y = data['bbsgt'].get_bbs_as_gt()

        rgbd = np.concatenate((x, depth), -1)
        transformed_rgbd, y = self._transform_batch(rgbd, y)
        x = transformed_rgbd[:3]
        depth = transformed_rgbd[-1].unsqueeze(0)
        size = x.shape[1:]

        return {
            'env': env,
            'episode': episode,
            'step': step,
            'image': x,
            "depth": depth,
            "location": torch.tensor(location),
            "instances": y,
            'width': size[1],
            'height': size[0],
            'info': f"env_{env}_episode_{episode}_step_{step}",
        }
        
        
class PseudoFullDataset(BbsgtDataset):
    '''
    对伪标签也进行处理比如transfrom
    '''
    def __init__(
        self,
        data_path,
        pseudo_labels,
        sampler=None,
        # consecutive_obs=1,
        subsample_factor=1,
        *args,
        **kwargs,
    ):

        if sampler is None:
            sampler = SampleLoader(data_path)
        (
            env_list,
            episode_list,
            steps_list,
        ) = sampler.get_env_episode_and_steps_dense_list(*args, **kwargs)
        
        # remove empty labels
        self.pseudo_labels = []
        mask = []
        for pseudo in pseudo_labels:
            if len(pseudo) > 0:
                self.pseudo_labels.append(pseudo)
                mask.append(True)
            else:
                mask.append(False)

        if len(episode_list) > len(pseudo_labels):      # TODO? 这里应该怎么改？
            mask += [False] * (len(episode_list) - len(pseudo_labels))

        steps_list = steps_list[mask]
        episode_list = episode_list[mask]
        pseudo_list = self.pseudo_labels
        
        inputs = np.array([x for x in zip(episode_list, steps_list, pseudo_list)])

        super().__init__(
            data_path=data_path,
            sampler=sampler,
            inputs=inputs,
            *args,
            **kwargs,
        )
    
    def __getitem__(self, idx):
        """
        """

        result = []
        episode, step, pseudo_label = self.inputs[self.index[idx]]

        camera_id = 0
        data = self.sampler.get_sample_multimodality(
            episode, camera_id, self.modalities, step
        )
        
        x = data['rgb'].data
        # y
        coco_ann = pseudo_label
        y = du.annotations_to_instances(
            coco_ann, x.shape[1:], mask_format='bitmask'
        )
        y.infos = [x['infos'] for x in coco_ann]
        y.gt_logits = torch.stack([x['gt_logits'] for x in coco_ann])
        if isinstance(y.gt_masks, BitMasks):
            y.gt_masks = y.gt_masks.tensor
            
        
        x, y = self._transform_batch_with_logits(x, y)   # TODO
        
        gt = data['bbsgt'].get_bbs_as_gt()
        
        size = x.shape[1:]
        return {        # TODO; 需要这么多吗
            'episode': episode,
            'image': x,
            'image_id': idx,
            'instances': y,
            "gt": gt,
            'width': size[1],
            'height': size[0],
        }
    
    def _transform_batch_with_logits(self, x, y, remap_classes=True):
        '''
        根据gt_logits进行变换？
        '''

        if isinstance(self.transform, ToTensorV2):
            out = self.transform(image=x)
            x = out['image']

        else:

            transformed = self.transform(
                image=x,
                bboxes=y.gt_boxes.tensor.numpy(),
                class_labels=y.gt_classes,
                masks=[m for m in y.gt_masks.numpy().astype(np.uint8)],
                infos=y.infos,
                gt_logits=[l for l in y.gt_logits],     # TODO：
            )

            x = transformed_image

            min_area = -1  # transform._to_dict()['bbox_params']['min_area']

            if len(transformed['class_labels']) > 0:

                semantic_masks_boxes = [cv2.boundingRect(x.numpy() if not isinstance(x, np.ndarray) else x) for x in transformed['masks']]
                semantic_masks_area = [x[-1] * x[-2] for x in semantic_masks_boxes]
                gt_masks = BitMasks(
                    torch.stack(
                        [
                            torch.tensor(x)
                            for x, area in zip(transformed['masks'], semantic_masks_area)
                            if area > min_area      # 有什么区别？TODO
                        ]
                    )
                )

                y = Instances(
                    image_size=x.shape[1:],
                    gt_boxes=Boxes(transformed['bboxes']),
                    gt_classes=torch.stack(transformed['class_labels']),
                    gt_logits=torch.stack(transformed['gt_logits']),
                    gt_masks=gt_masks,
                    infos=transformed['infos'],
                )
            else:
                y = Instances(
                    gt_boxes=Boxes(torch.Tensor()),
                    image_size=x.shape[1:],
                    gt_masks=BitMasks(torch.Tensor(size=[0, *x.shape[1:]])),
                    gt_classes=torch.Tensor(),
                    gt_logits=torch.Tensor(),
                    infos=[],
                )

        return x, y

if __name__ == "__main__":
    exp_p = '/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/test'+ "/episodes_data"
    dataset = BbsgtDataset(data_path=exp_p)
    print(dataset[4])
    dataset.get_coco_item_dict(4)
    
    dataset_full = FullDataset(data_path=exp_p)
    print(len(dataset_full))
    print(dataset_full[1])