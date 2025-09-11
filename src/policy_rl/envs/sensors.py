from habitat.core.registry import registry
import torch
import numpy as np
import habitat
from typing import Any
from gym import spaces
import logging
log = logging.getLogger(__name__)
import cv2
from detectron2.structures import Boxes, Instances
from src.vqf_constants import SIM_TO_COCO_MAPPING

@registry.register_sensor(name="object_detector_gt")
class ObjectDetectorGT(habitat.Sensor):
    MATTERPORT_SIM_TO_COCO_MAPPING = {
        5:60,
        3: 56,  # chair
        10: 57,  # couch
        14: 58,  # plan
        11: 59,  # beed
        18: 61,  # toilet
        22: 62,  # tv

    }
    
    # SIM_TO_COCO_MAPPING = {
    #     "chair": 56,  # chair
    #     "couch": 57,  # couch
    #     "potted plant": 58,  # plan
    #     "bed": 59,  # bed
    #     "toilet": 61,  # toilet
    #     # "tv": 62,  # tv
    #     # "dining table": 60,  # dining table
    # }
    SIM_TO_COCO_MAPPING = SIM_TO_COCO_MAPPING
    
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

        self._objects = self._sim.semantic_annotations().objects

        self.scene = ""

        # filter occluded objects in mp3d by checking avg depth inside instance mask wrt gt
        self.filter_occluded_instances = True
    
    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bbsgt"
    
    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT
    
    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )
    def _convert_matterport_to_coco_labels(self, label):
        switch = {
            'table': 'dining table',
            'plant': 'potted plant',
            'sofa': 'couch',
            'tv_monitor': 'tv',
        }
        if label in switch.keys():
            return switch[label]
        else:
            return label
        
    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        sense = kwargs['observations']['semantic']

        current_scene = self._sim.habitat_config.SCENE
        if current_scene != self.scene:

            self._objects = self._sim.semantic_annotations().objects
            self.mapping = {
                int(obj.id.split("_")[-1]): obj.category.name()
                for obj in self._sim.semantic_annotations().objects
                if obj is not None
            }
            self.scene = current_scene
            log.info("Updating mapping")

        bounding_boxes = []
        classes = []
        pred_masks = []
        infos = []
        mask_cleaned = sense.astype('uint8')

        objects_id = np.unique(mask_cleaned)
        for id_object in objects_id:

            bb = cv2.boundingRect((mask_cleaned == id_object).astype('uint8'))
            x, y, w, h = bb
            if id_object not in self.mapping:
                continue

            if (mask_cleaned == id_object).sum() < 1000:
                continue
            habitat_id = self._convert_matterport_to_coco_labels(
                self.mapping[id_object]
            )

            if habitat_id in self.SIM_TO_COCO_MAPPING:
                pred_mask = torch.zeros(sense.shape, dtype=torch.bool)
                pred_mask[(mask_cleaned == id_object).astype('bool')] = True

                coco_id = self.SIM_TO_COCO_MAPPING[habitat_id]
                bounding_boxes.append(torch.tensor([x, y, x + w, y + h]).unsqueeze(0))
                classes.append(torch.tensor(coco_id).unsqueeze(0))
                pred_masks.append(pred_mask.unsqueeze(0))
                infos.append(
                    {
                        'id_object': id_object,
                        'center': self._objects[id_object].aabb.center,
                    }
                )

        if len(bounding_boxes) > 0:
            results = Instances(
                image_size=sense.shape,
                pred_boxes=Boxes(torch.cat(bounding_boxes)),
                pred_classes=torch.cat(classes),
                scores=torch.ones(len(bounding_boxes)),
                pred_masks=torch.cat(pred_masks),
                infos=np.array(infos),
            )

        else:
            results = Instances(
                pred_boxes=Boxes(torch.Tensor()),
                image_size=sense.shape,
                pred_classes=torch.Tensor(),
                pred_masks=torch.Tensor(),
                scores=torch.Tensor(),
                infos=infos,
            )

        return {'instances': results}
    
@registry.register_sensor(name="position_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        return {
            'position': self._sim.get_agent_state().position,
            'orientation': self._sim.get_agent_state().rotation,
        }