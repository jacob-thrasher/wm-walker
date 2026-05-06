import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import env_utils



data = np.load('expert_data/walker/test/0.npz')


obs = data['obs']
print(obs.shape)

plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=1)
img = ax.imshow(obs[0])
plt.show()

for i in range(len(obs)):
    img.set_data(obs[i]) 
    fig.canvas.draw()
    fig.canvas.flush_events()

