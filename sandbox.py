import numpy as np
import matplotlib.pyplot as plt
import time


data = np.load('data/test.npz')

obs = data['obs']
ta = data['ta']
print(ta[0])
# print(obs.shape)

# plt.ion()
# fig, ax = plt.subplots()
# ax.imshow(obs[0])
# plt.show()
# for o in obs[1:]:
#     ax.imshow(o)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
