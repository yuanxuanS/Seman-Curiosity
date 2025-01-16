import pytorch_lightning as pl
import albumentations as A
import pickle
import os
import logging
from finetune.dataset import BbsgtDataset, FullDataset, PseudoFullDataset
from finetune.detector.augmentations import get_transform
from finetune.utils.train_helpers import dict_helper_collate, list_helper_collate
from finetune.dataset_utils import get_loader, get_coco_item_dict, SampleLoader

log = logging.getLogger(__name__)


class HabitatDataModule(pl.LightningDataModule):
    '''
    call: prepare_data()
          setup()
    '''
    def __init__(self, 
                 pseudo_labeler=None, 
                #  policy, 
                 dataset_path="", 
                 data_base_dir="", 
                 test_set="", 
                 transform_type='none', 
                 batch_size=8, 
                 val_batch_size=4,
                 *args, **kwargs ):
        super().__init__()

        self.pseudo_labeler = pseudo_labeler
        
        self.dataset_path = dataset_path
        self.data_base_dir = data_base_dir
        self.test_set = test_set
        
        self.transform_type  = transform_type
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        
        self.num_workers = 0    # TODO?
        
        
    
    def prepare_data(self):
        '''pl func, called by trainer
           prepare and save labels(coco type)
        '''
        sampler = self._get_sampler()
        labels = self._get_labels(sampler)
        with open("labels.pkl", "wb") as fp:
            pickle.dump(labels, fp)

    def _get_sampler(self):
        sampler = SampleLoader(self.dataset_path)
        return sampler
    
    def _get_labels(self, sampler):
        val_transform = A.Compose(
            get_transform("none"),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['class_labels', 'infos'],
            ),
        )
        pseudolabel_dataset = FullDataset(
            None,
            sampler=sampler,
            transform=val_transform,
        )

        pseudolabel_loader = get_loader(
            pseudolabel_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=dict_helper_collate,
        )
        
        pseudolabel_trainer = pl.Trainer(gpus=1)
        model_outs = pseudolabel_trainer.predict(
            self.pseudo_labeler, pseudolabel_loader
        )
        pseudo_labels = self.pseudo_labeler.get_pseudo_labels(
            model_outs, pseudolabel_loader
        )

        coco_pseudo_labels = get_coco_item_dict(pseudo_labels)  # 必须coco格式？

        return coco_pseudo_labels
    
    def setup(self):
        '''pl func, called by trainer
            get datasets
        '''
        sampler = self._get_sampler()

        with open('labels.pkl', 'rb') as handle:
            labels = pickle.load(handle)

        self.train_dataset = self._get_dataset(sampler, labels)
        self.test_dataset = self._get_validation()      # 指最后的test_set
        
    def train_dataloader(self):
        '''pl func'''
        train_loader = get_loader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_helper_collate,
        )
        return train_loader
    
    def val_dataloader(self):
        '''pl func'''
        test_loader = get_loader(
                self.test_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=dict_helper_collate,
            )

        return test_loader
    
    def test_dataloader(self):
        '''pl func'''
        test_loader = get_loader(
                self.test_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=dict_helper_collate,
            )

        return test_loader

    def _get_dataset(self, sampler, coco_pseudo_labels):
        """
        ## TODO ?Apply pseudo-labeler and return consistent pseudolabel dataset
        """

        train_transform = A.Compose(
            get_transform(self.transform_type),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos', 'gt_logits'],
            ),
        )
        assert len(coco_pseudo_labels) > 0, "No pseudo-labels provided"
        assert len(coco_pseudo_labels) == len(
            sampler
        ), f"Expected {len(sampler)} got {len(coco_pseudo_labels)}"

        train_dataset = PseudoFullDataset(
            data_path=None,
            pseudo_labels=coco_pseudo_labels,
            sampler=sampler,
            transform=train_transform,  # TODO
            # consecutive_obs=self.consecutive_obs
        )
        return train_dataset
    
    def _get_validation(self):
        transform = A.Compose(
            get_transform("none"),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels', 'infos'],
            ),
        )

        dataset = BbsgtDataset(
            os.path.join(self.data_base_dir, self.test_set),
            transform=transform,
        )
        return dataset
        
class GTDataModule(HabitatDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_data(self):
        ''' no need to implement'''
        pass
    
    def setup(self, stage):
        sampler = self._get_sampler()

        # no need to load labels
        
        self.train_dataset = self._get_dataset(sampler)
        self.test_dataset = self._get_validation()
        
    def _get_dataset(self, sampler: SampleLoader):
        ''' get bbsgt to construct dataset'''
        log.info("Using ground-truth for detector training")
        
        train_transform = A.Compose(
            get_transform(self.transform_type),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos'],
            ),
        )
        
        inputs = sampler.get_env_episode_and_steps_dense_list()     
        filter_empty_instances = []
        
        for env, ep, step in zip(inputs[0], inputs[1], inputs[2]):
            instances = sampler.get_sample(env, ep, step, "bbsgt").get_bbs_as_gt()

            filter_empty_instances.append(len(instances) > 0)       # 仅保留有mask的


        return FullDataset(
            data_path=None,
            sampler=sampler,
            index_mask=filter_empty_instances,      # 仅保留有mask的
            transform=train_transform,
        )
        
if __name__ == "__main__":
    p = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/test" + "/episodes_data"
    
    dm = GTDataModule(dataset_path=p)
    dm.setup("train")