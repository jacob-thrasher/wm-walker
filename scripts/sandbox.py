import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)



data = np.load('expert_data/walker/test/0.npz')


obs = data['obs']
print(obs.shape)
print(data['ta'].shape)
print(data.keys())
# print(obs.size())
# print(obs.shape)

# plt.ion()
# fig, ax = plt.subplots()
# ax.imshow(obs[0])
# plt.show()
# for o in obs[1:]:
#     ax.imshow(o)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
