# test_make_world.py
from multiagent.scenarios.Reilaygh_scenario import Scenario

scenario = Scenario()
world = scenario.make_world()
for agent in world.agents:
    print(f"Agent {agent.name} t {agent.thickness} v {agent.velocity}")
    reward = scenario.reward(agent,world)
    print(f"Reward: {reward}")
    observation = scenario.observation(agent,world)
    print(f"Observation: {observation}")
print(f"Number of agents: {len(world.agents)}")
print(f"Agent names: {[a.name for a in world.agents]}")
print(f"Landmark count: {len(world.landmarks)}")
