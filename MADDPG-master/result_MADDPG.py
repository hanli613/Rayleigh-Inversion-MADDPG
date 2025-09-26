import numpy as np
import matplotlib.pyplot as plt

# 加载 .npy 文件
data = np.load('model/simple_num/returns.pkl.npy')


# 定义映射函数
def map_range(data, original_min, original_max, target_min, target_max):
    return ((data - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min

# 应用映射函数
mapped_data = map_range(data, -4000, -300, 0, 500)
print(np.shape(mapped_data))
print(mapped_data)
# 绘制图像
plt.plot(mapped_data)
plt.xlabel('Episode')
plt.ylabel('Maddpg Reward')
plt.title('Reward')
plt.show()
