import pytorch_lightning as pl
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from torchmetrics.detection.map import MAP

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
        args.confidence_threshold
    cfg.freeze()
    return cfg

class Predictor(pl.LightningModule):
    def __init__(self, cfg=None, 
                 input_format=None, 
                 load_checkpoint=True, 
                 metadata=None, 
                 ):
        super().__init__()
        
        cfg = setup_cfg(cfg)
        self.cfg = cfg.clone()
        
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        
        self.test_map_metric = MAP(class_metrics=True)
    
    def on_test_epoch_end(self):
        pass
    
    def test_step(self):
        pass
    
    def forward(self):
        pass
    
    def infer(self):
        pass
    
    
        