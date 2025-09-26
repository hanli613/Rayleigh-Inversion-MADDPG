import numpy as np
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.true_model = {
            'layer_0': {'thickness': 1.2, 'velocity': 220},
            'layer_1': {'thickness': 2.0, 'velocity': 300},
            'layer_2': {'thickness': 1.5, 'velocity': 250},
            'layer_3': {'thickness': 2.5, 'velocity': 350},
            'layer_4': {'thickness': 1.0, 'velocity': 180},
        }

    def make_world(self):
        world = World()
        # 设置世界属性
        world.dim_c = 0
        num_agents = 5  # 修改为 5 个智能体
        world.collaborative = True
        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
            # 新增：每层的物理参数
            agent.thickness = 1.0  # 初始层厚
            agent.velocity = 300.0  # 初始波速
        # 初始化条件
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # 设置颜色（不同颜色区分智能体和目标）
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])  # 蓝色系

        # 设置智能体和目标的随机初始位置
        # 初始化真实模型（假设有 5 层）
        world.num_layers = 5
        world.true_model = [
            {'thickness': 1.2, 'velocity': 220},
            {'thickness': 2.0, 'velocity': 300},
            {'thickness': 1.5, 'velocity': 250},
            {'thickness': 2.5, 'velocity': 350},
            {'thickness': 1.0, 'velocity': 180}
        ]

        # 将真实值赋给各 Agent
        for i, agent in enumerate(world.agents):
            agent.thickness = np.random.uniform(0.5, 3.0)
            agent.velocity = np.random.uniform(150, 400)

    def benchmark_data(self, agent, world):
        rew = 0
        thickness_error = 0.0
        velocity_error = 0.0
        layer_idx = int(agent.name.split('_')[1])  # 提取编号

        # 计算误差
        true_thickness = self.true_model[f'layer_{layer_idx}']['thickness']
        true_velocity = self.true_model[f'layer_{layer_idx}']['velocity']

        thickness_error += abs(agent.thickness - true_thickness)
        velocity_error += abs(agent.velocity - true_velocity)

        return thickness_error, velocity_error

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        layer_idx = int(agent.name.split('_')[1])  # 提取编号

        # 获取当前层的真实值
        true_t = self.true_model[f'layer_{layer_idx}']['thickness']
        true_v = self.true_model[f'layer_{layer_idx}']['velocity']

        # 当前预测值
        pred_t = agent.thickness
        pred_v = agent.velocity

        # 奖励函数（越接近越好）
        rew = -(abs(pred_t - true_t) + 0.5 * abs(pred_v - true_v))

        return rew

    def observation(self, agent, world):
        layer_idx = int(agent.name.split('_')[1])  # 提取编号
        own_t = agent.thickness
        own_v = agent.velocity

        # 获取当前层的真实值
        true_t = self.true_model[f'layer_{layer_idx}']['thickness']
        true_v = self.true_model[f'layer_{layer_idx}']['velocity']

        # 目标参数（相对误差）
        target_t = true_t - own_t
        target_v = true_v - own_v

        # 其他智能体的参数（相对当前智能体）
        other_t = []
        other_v = []
        for other in world.agents:
            if other is agent:
                continue
            other_t.append(other.thickness - own_t)  # 注意这里是 other.thickness
            other_v.append(other.velocity - own_v)  # 同理

        # 拼接观测空间
        return np.concatenate([
            np.array([own_t, own_v, target_t, target_v]),
            np.array(other_t),
            np.array(other_v)
        ])





