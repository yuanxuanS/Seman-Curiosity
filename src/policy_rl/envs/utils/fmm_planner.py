import cv2
import numpy as np
import skfmm
import skimage
from numpy import ma
import random

def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2 \
               and ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 > \
                    (step_size - 1) ** 2:
                mask[i, j] = 1

    mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2:
                mask[i, j] = max(5,
                                 (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                  ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    def __init__(self, traversible, scale=1, step_size=5):
        self.scale = scale
        self.step_size = step_size
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))
        self.fmm_dist = None

    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
            int(goal[1] / (self.scale * 1.))

        if self.traversible[goal_x, goal_y] == 0. and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)   # 计算等高线，值为距离
        dd = ma.filled(dd, np.max(dd) + 1)      # 将False区域都赋值为最大值
        self.fmm_dist = dd
        return

    def set_multi_goal(self, goal_map):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)  # 可通行1, 其他0
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)   # 计算等高线，值为距离，到目标的最短路径距离
        dd = ma.filled(dd, np.max(dd) + 1)  # mask的区域比如障碍物填充大值
        self.fmm_dist = dd
        return

    def get_short_term_goal(self, state):   # state位置点，找到state周围成本最低的可行点
        scale = self.scale * 1.
        state = [x / scale for x in state]      # 缩放位置坐标
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1]) 
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)  # 填充不可达区域
        subset = dist[state[0]:state[0] + 2 * self.du + 1,     # 截取当前state周围
                      state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
            subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2      # mask区域*大数，设被掩码区域的值很高

        if subset[self.du, self.du] < 0.25 * 100 / 5.:  # 25cm  中心点的值是否小于25cm
            stop = True     # 到达目标点
        else:
            stop = False

        subset -= subset[self.du, self.du]      # 减去中心值，小于0的负数，越小代表距离目标越近/等高图上距离goal越近
        ratio1 = subset / dist_mask     # 计算成本？
        subset[ratio1 < -1.5] = 1   # 成本小于-1.5， 作为可行区域

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)      # 最小
        # min_values = np.min(subset)
        # indices = np.where(subset == min_values)
        # if len(indices) >0:
        #     if len(indices[0]) > 1: # 多个最小值
        #         idx = random.choice([i for i in range(len(indices[0]))])
        #         stg_x, stg_y = indices[0][idx], indices[1][idx]
        #     else:
        #         stg_x, stg_y = indices
            
        
        if subset[stg_x, stg_y] > -0.0001:
            replan = True       # 选择的stg就是目标点
        else:
            replan = False

        return (stg_x + state[0] - self.du) * scale, \
               (stg_y + state[1] - self.du) * scale, replan, stop

    def _find_nearest_goal(self, goal):
        traversible = skimage.morphology.binary_dilation(
            np.zeros(self.traversible.shape),
            skimage.morphology.disk(2)) != True
        traversible = traversible * 1.
        planner = FMMPlanner(traversible)
        planner.set_goal(goal)

        mask = self.traversible

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal
