import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4])
seen_loss = [0.04734128984835064, 0.032016380530733024, 0.026222193664707766, 0.02245828090235591]
y1 = np.array(seen_loss)
unseen_loss = [0.15347770869858185, 0.15297774370696585, 0.1534134098565427, 0.15371372599659835]
y2 = np.array(unseen_loss)

real_loss = [0.054396767287578876, 0.054129540667472775, 0.05361355341918287, 0.05364461250421756]
y3 = np.array(real_loss)
plt.plot(x, y1, label='test seen loss')  # 第一条曲线
plt.plot(x, y2, label='test unseen loss')  # 第二条曲线
plt.plot(x, y3, label='test real loss')  # 第二条曲线
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.show()
plt.savefig('test_plot.png')