from src.policy_rl.agents.utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
from src.policy_rl.arguments import get_args
import cv2

if __name__ == "__main__":
    
    # 读取
    dir_pth = ""
    rgb = cv2.imread(dir_pth)
    
    args = get_args()
    sem_pred = SemanticPredMaskRCNN(args)
    
    # predict by maskrcnn
    semantic_pred, rgb_vis, instance = sem_pred.get_prediction(rgb, return_score=False, return_instance=True)
    if len(instance) == 0:
        return 1