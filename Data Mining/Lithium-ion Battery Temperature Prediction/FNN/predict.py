import matplotlib.pyplot as plt
import numpy as np
from train import MAE_list


y1, y2 = zip(*MAE_list)
x = np.arange(0, len(y1))

plt.plot(x, y1, label='train')
plt.plot(x, y2, label='valid')
plt.xlabel("epoch")
plt.ylabel("Loss: MAE")
plt.legend()
plt.show()