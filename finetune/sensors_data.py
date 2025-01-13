import abc
import cv2
from finetune.sensors_utils import SenseInfo, get_sense_info
from dataclasses import dataclass
import numpy as np

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
            self.name = f"{self.sense_info.episode}-{self.sense_info.mod}-{self.sense_info.camera_id}"
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

MODALITY_SENSE = {
    "rgb": VisualSense, #RGBSense,
    "depth":  VisualSense,  #DepthSense,
    "semantic": VisualSense,    #SemanticSense,
    "semanticinstances": VisualSense,   #SemanticInstancesSense,
    "bbs": VisualSense, #BBPredSense,
    "bbsgt": VisualSense,#    BBSense,
    'position': VisualSense,    #AgentPoseSense,
    'egomap': VisualSense,  #EgomapSense,
    'disagreement_map': VisualSense,    #DisagreementSense,
    "map_sensor": VisualSense,  #TopDownMapSense,
}