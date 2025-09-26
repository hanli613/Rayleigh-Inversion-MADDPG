import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # 设置世界属性
        world.dim_c = 2
        num_agents = 5  # 修改为 5 个智能体
        num_landmarks = 5  # 修改为 5 个目标
        world.collaborative = True
        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # 添加地标（目标）
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'target_{i}'
            landmark.collide = False
            landmark.movable = False
        # 初始化条件
        self.reset_world(world)
        return world
    def reset_world(self, world):
        # 设置颜色（不同颜色区分智能体和目标）
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])  # 蓝色系
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.25, 0.25])  # 红色系

        # 设置智能体和目标的随机初始位置
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        """
        每个 Agent 只对其编号相同的 Landmark 负责
        """
        agent_idx = int(agent.name.split('_')[1])  # 提取编号
        target = world.landmarks[agent_idx]  # 对应的目标

        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos)))
        rew = -dist  # 距离越小越好

        # 如果发生碰撞也惩罚
        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1

        return rew

    def observation(self, agent, world):
        agent_idx = int(agent.name.split('_')[1])
        target = world.landmarks[agent_idx]

        # 自身状态
        own_pos = agent.state.p_pos
        own_vel = agent.state.p_vel

        # 目标位置（相对坐标）
        target_pos = target.state.p_pos - agent.state.p_pos

        # 其他智能体的位置（相对）
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([own_vel, own_pos, target_pos, *other_pos])
