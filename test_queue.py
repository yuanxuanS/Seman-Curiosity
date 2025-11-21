import numpy as np
from collections import deque, defaultdict
import heapq
from typing import Dict, Any, List, Tuple


class TemporalPrioritySequences:
    def __init__(self, T: int=10, B: int=10):
        """
        初始化每个物体的时序优先队列
        :param B: 队列容量（记录T步内的信号）
        heap: List[deque(), ...] 多个orient序列, 根据confidence排序
        """
        self.T = T
        self.B = B # 存储最大序列条数
        self.timestep = 0  # 当前时间步
        self.sequence_heap = []      # 按置信度排序的最小堆 List[(-confidence, deque), (), ...]
        self.sequence_confidence = []   # 置信度列表
        # self.queue = deque(maxlen=T)  # 按时间顺序存储信号
        self.thres = 30     # 判断预测和理论上推断之间的差距
    
    def compute_ori_by_action(self, orient: int, action):
        '''
        根据action和orient计算新的orient
        orient: int, 0-359
        action: int, 0: forward, 1:left, 2: right
        '''
        if action == 0:
            new_orient = orient
        elif action == 1:
            new_orient = (orient + 30) % 360
        elif action == 2:
            new_orient = (orient - 30)  % 360  
        return new_orient
    
    def update_all_orient_in_queue(self, action: int, orient_pred, confidence: float,):
        '''
        对队列中存储的所有序列，用上一时刻 orient和action预测此刻的orient; 如果没有任一序列的orient和orient_pred的误差在范围内，则作为新序列
        orient_pred: OriAnything 预测的朝向
        '''
        been_added = True
        for i, data in enumerate(self.sequence_heap):       
            confidence_orig, sequence_queue = data
            orient_last = sequence_queue[-1]
            orient_gt = self.compute_ori_by_action(orient_last, action)
            if abs(orient_gt - orient_pred) > self.thres: # 误差过大，作为新序列
                pass       # 该置信度随步数减少？
            else:
                been_added = False      # 只要orient_pred有一个序列在误差范围内，就不作为新序列
                orient_new = orient_pred        # 可以取 orient_pred + orient_this / 2
                self.update_sequence(sequence_queue, orient_new)
                data[0] -= confidence       # 置信度更新方式有待测试
        
        if been_added:
            sequence = self.create_sequence(orient_pred) 
            self.add_new_sequence(confidence, sequence)              
    
    def create_sequence(self, orient):
        sequence = deque(maxlen=self.T)
        sequence.append(orient)
        return sequence
    
    def update_sequence(self, sequence: deque, orient: float):
        '''
        更新orient到queue末尾；注意最大序列长度T
        '''
        sequence.append(orient)
        
    def add_new_sequence(self, confidence: float, data: deque):
        """
        添加新orient序列到队列
        :param confidence: 置信度（用于排序）
        :param data: 信号数据
        
        如果队列已满，排序后移除置信度最低的
        """
        # 生成时间戳标识
        # self.timestep = (self.timestep + 1) % self.T
        # timestamp = self.timestep
        
        # 创建信号条目（置信度、时间戳、数据）
        signal = [-confidence, data]  # 负置信度实现最大堆
        
        # 队列已满时移除最旧信号
        if len(self.sequence_heap) == self.B:
            # 从堆中移除旧信号（惰性删除）
            heapq.heappush(self.sequence_heap, signal)
            min_confidence, min_data = heapq.heappop(self.sequence_heap)
        else:
            heapq.heappush(self.sequence_heap, signal)
            
    
class CategoryObjects:
    def __init__(self, T:int =10):
        """
        初始化类别对象管理器
        每个物体存储中心位置
        Categoryobjects: "cls": List[(obj_center, obj_queue)]
        """
        self.T = T  # 队列的最大长度
        self.Categoryobjects: Dict[str: List] = {}
    
    def _new_class(self, category: str, object_data):
        """
        新建类别
        object_data: 包括物体中心、信息; (center, obj_priorqueue_info)
        obj_priorqueue_info: (orient, confidence)
        """
        center, obj_priorqueue_info = object_data
        action, orient, confidence = obj_priorqueue_info
        
        sequences = TemporalPrioritySequences()
        sequence = sequences.create_sequence(orient)
        sequences.add_new_sequence(confidence, sequence)
        self.Categoryobjects[category] = [[center, sequences]]
    
    def _new_object(self, category: str, object_data):
        """
        新建已有类别，新物体的队列
        """
        center, obj_priorqueue_info = object_data
        action, orient, confidence = obj_priorqueue_info
        
        sequences = TemporalPrioritySequences()
        sequence = sequences.create_sequence(orient)
        sequences.add_new_sequence(confidence, sequence)
        self.Categoryobjects[category].append([center, sequences])
        
    def update_info(self, obj_center, obj_feature, obj_category, obj_priorqueue_info):
        '''
        判断是否为新类别、新物体
        
        '''
        object_data = (obj_center, obj_priorqueue_info)
        if obj_category not in self.Categoryobjects:        # 新类别
            self._new_class(obj_category, object_data)
        else:
            # 判断是否为新物体: 空间信息、外观信息
            is_new, object_id = self._recognize_object(obj_center)
            if is_new:
                self._new_object(obj_category, object_data)  # 
            else:
                self._update_object(object_data, object_id)
                
    def _update_object(self, object_data, object_id):
        '''
        更新物体信息到已有object
        object_data: (object_center, object_priorqueue_info)
        object_id : (category, object_list_idx)
        '''
        object_center, object_queue_info = object_data
        category, object_list_idx = object_id
        object_center_orig, object_priorqueue =  self.Categoryobjects[category][object_list_idx]
        # update center
        object_center_new = (object_center_orig + object_center) / 2
        # update queue info
        action, orient, confidence = object_queue_info
        object_priorqueue.update_all_orient_in_queue(action, orient, confidence)
        self.Categoryobjects[category][object_list_idx] = [object_center_new, object_priorqueue]
        
                
    def _recognize_object(self, obj_center):
        
        for class_str, objects in self.Categoryobjects.items():
            for n, object in enumerate(objects):
                obj_ct, _ = object
                if euclidean_distance(obj_center, obj_ct) < 0.5:         # 距离阈值
                    return False, (class_str, n)
        return True, None
    
    
def euclidean_distance(x, y):
    """
    计算两个向量的欧式距离
    """
    return np.linalg.norm(x - y)  # 直接计算 L2 范数


if __name__ == "__main__":
    
    sequences = TemporalPrioritySequences()
    
    CateObjs = CategoryObjects()
    
    # 新类
    obj_center = np.array([1, 2])   # RGB+depth
    obj_feature = np.array([1, 2, 3])   
    obj_category = "chair"      # detector
    orient = 30
    confidence = 1.5
    obj_priorqueue_info = [None, orient, confidence]     
    CateObjs.update_info(obj_center, obj_feature, obj_category, obj_priorqueue_info)
    
    # 同类+不同物
    obj_center1 = np.array([2,3])
    obj_feature1 = None
    obj_category1 = "chair"
    orient1 = 90
    confidence1 = 1.3
    obj_priorqueue_info1 = [0, orient1, confidence1]
    CateObjs.update_info(obj_center1, obj_feature1, obj_category1, obj_priorqueue_info1)
    
    # 同类，同物, 误差在范围内
    obj_center2 = np.array([2,3])
    obj_feature2 = None
    obj_category2 = "chair"
    orient2 = 100
    confidence2 = 1.4
    obj_priorqueue_info2 = [1, orient2, confidence2]
    CateObjs.update_info(obj_center2, obj_feature2, obj_category2, obj_priorqueue_info2)
    
    # 同类，同物， 误差不在范围内
    obj_center3 = np.array([2,3])
    obj_feature3 = None
    obj_category3 = "chair"
    orient3 = 150
    confidence3 = 1.4
    obj_priorqueue_info3 = [0, orient3, confidence3]
    CateObjs.update_info(obj_center3, obj_feature3, obj_category3, obj_priorqueue_info3)
    
