import abc
import cv2
from src.finetune.sensors_utils import SenseInfo, get_sense_info
from src.constants import coco_categories_mapping
from dataclasses import dataclass
import numpy as np
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import torch
import dataclasses
import quaternion

WIDTH, HEIGHT = 300, 400
@dataclass
class Intrinsics:
    xc: float
    yc: float
    focal_length: float
    width: int
    height: int

    def get_mat(self) -> np.ndarray:
        return np.array(
            [
                [self.focal_length, 0, self.xc],
                [0.0, self.focal_length, self.yc],
                [0.0, 0, 1],
            ]
        )
        
class Sense(abc.ABC):
    def __init__(self, path: str = None, sense_info: SenseInfo = None):
        if sense_info is None and path is not None:
            self.sense_info = get_sense_info(path)
        elif sense_info is not None:
            self.sense_info = sense_info
        else:
            self.sense_info = None

        if self.sense_info is not None:
            self.name = f"{self.sense_info.env_id}-{self.sense_info.episode}-{self.sense_info.step}-{self.sense_info.mod}"
        else:
            self.name = ""

    @staticmethod
    def load(path):
        return Sense(path)
    
class VisualSense(Sense):
    HFOV_DEG = 90

    def get_camera_matrix(self, fov=HFOV_DEG):
        """
        From Object-Goal-Navigation
        Returns a camera matrix from image size and fov.
        """
        width = height = self.get_width()
        xc = (width - 1.0) / 2.0
        yc = (height - 1.0) / 2.0
        f = (width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)

        return Intrinsics(xc, yc, f, width, height)

    def __init__(self, data: np.ndarray = None, path=None, sense_info=None):
        super().__init__(path, sense_info)

        self.data = data

    def get_width(self):
        return self.data.shape[0]

    def show(self):
        cv2.imshow(self.name, self.data)

    def get_image(self):
        return self.data



class RGBSense(VisualSense):
    CODE = "rgb"
    INPUT_FORM = "RGB"  # RGB

    def __init__(self, data: np.ndarray = None, path=None, sense_info=None):
        super().__init__(data, path, sense_info)

    @staticmethod
    def load(path):

        rgb_image = np.load(path)

        if (
            rgb_image.shape[0] == 3
            or rgb_image.shape[0] == 1
            or rgb_image.shape[0] == 4
        ):
            # channel-last
            rgb_image = rgb_image.transpose(1, 2, 0)

        if rgb_image.shape[-1] > 3:
            rgb_image = rgb_image[:, :, :-1]  # remove `a`` channel

        rgb_image = np.ascontiguousarray(
            rgb_image[:, :, ::-1]
        )  # RGB (from np loading) to BGR (cv2)

        return RGBSense(rgb_image, path)

class DepthSense(VisualSense):
    CODE = "depth"

    def __init__(self, data=None, path=None, sense_info=None):
        super().__init__(data, path, sense_info)

    @staticmethod
    def load(path):
        depth_image = np.load(path)

        if "neuralslam" in path:
            depth_image = depth_image * 10  # only for neuralslam

        return DepthSense(depth_image, path)

class SemanticSense(VisualSense):
    CODE = "semantic"

    def __init__(self, data: np.ndarray, path=None, sensor_info=None):
        super().__init__(data, path, sensor_info)

    @staticmethod
    def load(path):

        semantic_image = np.load(path).astype("uint8")
        return SemanticSense(semantic_image, path)

    def show(self):
        heatmap = self.get_image()
        cv2.imshow(self.name, heatmap)
    
    def get_image(self):
        return cv2.applyColorMap(self.data, cv2.COLORMAP_HSV)
    

class BBSense(VisualSense):
    CODE = "bbs"
    CLASSES = {
        56: "chair",
        57: "couch",
        58: "plant",
        59: "bed",
        61: "toilet",
        # 62: "tv",
        # 60: "table",
    }
    
    assert  CLASSES.keys() == coco_categories_mapping.keys()
    CLASSES_CLSAG = {
        0: "object",
    }
    REMAP = {i: k for i, k in enumerate(CLASSES)}
    CLASSES_TO_IDX = {k: i for i, k in enumerate(CLASSES.keys())}

    def __init__(self, bbs: Instances, frame=None, path=None, sense_info=None):
        super().__init__(bbs, path, sense_info)
        self.bbs = bbs
        rgb_sense_info = dataclasses.replace(self.sense_info, mod=RGBSense.CODE)

        try:
            if frame is None and rgb_sense_info is not None:
                frame = RGBSense.load(rgb_sense_info.get_path())
            self.frame = frame
        except Exception as ex:
            self.frame = None

    @staticmethod
    def load(path_bb):
        bbs = BBSense._load_bbs(path_bb)

        if len(bbs) > 0:
            mask = [x.item() in BBSense.CLASSES.keys() for x in bbs.pred_classes]
            bbs = bbs[mask]

        return BBSense(path=path_bb, bbs=bbs)

    @staticmethod
    def _load_bbs(path):
        raw_prediction = np.load(path, allow_pickle=True)

        instances = raw_prediction.item()['instances']
        if len(instances) == 0:
            return instances
        mask = [
            instances[i].pred_classes.item() in BBSense.CLASSES.keys()
            for i in range(len(instances))
        ]

        return instances[mask]

    def get_bbs_as_gt(self):

        target = Instances(self.bbs.image_size)
        target.gt_boxes = self.bbs.pred_boxes
        target.gt_classes = self.bbs.pred_classes

        if hasattr(self.bbs, "pred_masks"):
            target.gt_masks = self.bbs.pred_masks

        if hasattr(self.bbs, "infos"):

            target.infos = self.bbs.infos
            for t in target.infos:
                t['episode'] = self.sense_info.episode

        return target

    def get_bounding_boxes(self):
        if 'pred_boxes' in self.bbs:
            return self.bbs.pred_boxes
        else:
            return []

    def show(self):
        img = self.get_image()
        cv2.imshow(self.name, img)

        # cv2.imwrite(
        #     self.name + str(self.sense_info.step) + '.png',
        #     frame.get_image()[:, :, ::-1],
        # )

    def get_image(self):
        metadata = MetadataCatalog.get('coco_2017_val')
        visualizer = Visualizer(
            self.frame.data, metadata, instance_mode=ColorMode.IMAGE
        )
        self.bbs.scores = torch.tensor([0.99 for _ in range(len(self.bbs))])
        frame = visualizer.draw_instance_predictions(predictions=self.bbs.to('cpu'))
        img = cv2.resize(frame.get_image()[:, :, ::-1], (WIDTH, HEIGHT))
        return img

    def has_object(self):
        if len(self.bbs) > 0:
            return True
        return False

    def object_class(self):
        return self.bbs.pred_classes
    
class BBPredSense(BBSense):
    '''
    class id is start from 0; remap to orignal idx in cocodataset
    '''
    CODE = "bbs_prediction"
    CLASSES = {0: "couch",     # bbs prediction
        1: "plant",
        2: "bed",
        3: "toilet",
        4: "tv",
        5: "table",
    }
    REMAP = {i: k for i, k in enumerate(CLASSES)}
    # CLASSES_TO_IDX = {k: i for i, k in enumerate(CLASSES.keys())}
    CLS_ID_MAP = {0: 57, 1:58, 2:59, 3:61, 4:62, 5:60, 6:6}
    def __init__(self, bbs: Instances, frame=None, path=None, sense_info=None):
        super().__init__(bbs, frame, path, sense_info)

    @staticmethod
    def load(path_bb):
        bbs = BBPredSense._load_bbs(path_bb)

        if len(bbs) > 0:
            mask = [x.item() in BBPredSense.CLASSES.keys() for x in bbs.pred_classes]
            bbs = bbs[mask]

        return BBPredSense(path=path_bb, bbs=bbs)

    @staticmethod
    def _load_bbs(path):
        raw_prediction = np.load(path, allow_pickle=True)

        instances = raw_prediction.item()['instances']
        if len(instances) == 0:
            return instances
        mask = [
            instances[i].pred_classes.item() in BBPredSense.CLASSES.keys()
            for i in range(len(instances))
        ]

        return instances[mask]
    
    def get_image(self):
        metadata = MetadataCatalog.get('coco_2017_val')
        visualizer = Visualizer(
            self.frame.data, metadata, instance_mode=ColorMode.IMAGE
        )
        self.bbs.pred_classes = torch.tensor([BBPredSense.CLS_ID_MAP[int(i)] for i in self.bbs.pred_classes])
        frame = visualizer.draw_instance_predictions(predictions=self.bbs.to('cpu'))
        img = cv2.resize(frame.get_image()[:, :, ::-1], (WIDTH, HEIGHT))
        return img
    
    def save_img(self, save_pth, step, env_id):
        metadata = MetadataCatalog.get('coco_2017_val')
        v1 = Visualizer(
            self.frame.data, metadata, instance_mode=ColorMode.IMAGE
        )
        # self.bbs.pred_classes = torch.tensor([BBPredSense.CLS_ID_MAP[int(i)] for i in self.bbs.pred_classes])
        v1 = v1.draw_instance_predictions(predictions=self.bbs.to('cpu'))
        img = cv2.cvtColor(v1.get_image(), cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_pth+"Env "+str(step)+"_t_"+str(env_id) +" maskrcnn.png", img)

class Pose(Sense):
    AGENT_TO_SENSOR_TRANSLATION = np.array([0, 0.88, 0])

    def __init__(
        self,
        position: np.ndarray,
        orientation,
        reference: str,
        path=None,
        sense_info=None,
    ):
        super().__init__(path, sense_info)
        self.position = position
        self.orientation = orientation
        self.reference = reference

    def get_T(self):
        """
        Get pose_world transformation matrix for pose
        """
        rotation_0 = quaternion.as_rotation_matrix(self.orientation)
        T = np.eye(4)
        T[0:3, 0:3] = rotation_0
        T[0:3, 3] = self.position
        return T

    def get_transformation_to_pose(self, pose2):
        """
        Transformation from current pose to given pose
        """

        T_world_pose1 = self.get_T()
        T_world_pose2 = pose2.get_T()

        T_pose2_world = np.linalg.inv(T_world_pose2)

        T_pose2_pose1 = np.matmul(T_pose2_world, T_world_pose1)
        return T_pose2_pose1

class CamPoseSense(Pose):
    def __init__(
        self, position: np.ndarray, orientation: quaternion, path=None, sense_info=None
    ):
        super().__init__(position, orientation, "cam", path=path, sense_info=sense_info)



class AgentPoseSense(Pose):

    CODE = "position"

    def __init__(
        self, position: np.ndarray, orientation: quaternion, path=None, sense_info=None
    ):
        super().__init__(
            position, orientation, "agent", path=path, sense_info=sense_info
        )

    def get_T_world_agent(self):
        """
        Get pose_world transformation matrix for pose
        """
        rotation_0 = quaternion.as_rotation_matrix(self.orientation)
        T = np.eye(4)
        T[0:3, 0:3] = rotation_0
        T[0:3, 3] = self.position
        return T

    def get_cam_pose(self):
        """
        Get pose_world transformation matrix for pose
        """
        rot_mat = quaternion.as_rotation_matrix(self.orientation)
        translation = np.matmul(rot_mat, AgentPoseSense.AGENT_TO_SENSOR_TRANSLATION)
        position = self.position + translation
        return CamPoseSense(
            position=position, orientation=self.orientation, sense_info=self.sense_info
        )

    @staticmethod
    def load(path):

        location_data = np.load(path, allow_pickle=True)

        try:
            position = location_data.item()['position']
            orientation = location_data.item()['orientation']

        except Exception as ex:  # type: ignore[F841]
            position = location_data[0]
            orientation = location_data[1]

        return AgentPoseSense(position, orientation, path).get_cam_pose()
    


    
    
MODALITY_SENSE = {
    "rgb": RGBSense,
    "depth":  DepthSense,
    "semantic": SemanticSense,
    # "semanticinstances": VisualSense,   #SemanticInstancesSense,
    "bbs": BBPredSense,
    "bbsgt": BBSense,
    'position': AgentPoseSense,
    # 'egomap': VisualSense,  #EgomapSense,
    # 'disagreement_map': VisualSense,    #DisagreementSense,
    # "map_sensor": VisualSense,  #TopDownMapSense,
}