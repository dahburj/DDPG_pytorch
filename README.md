# DDPG_pytorch
Deep Deterministic Policy Gradient implementation in pytorch, adjusted to Unity environments

## Example usage
```python
from agent import Agent, Config
from teacher import Teacher

# Creating env
env = UnityEnvironment(file_name=UNITY_ENV)

# Getting env info
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

# Preparing agent for training
config = Config()
config.BUFFER_SIZE=int(3e4)

agent = Agent(state_size, action_size, config)
teacher = Teacher(agent, env, brain_name, num_agents)

# Training
res = teacher.train(1000, 100, 0.5)

# Display training results
Teacher.visualise_scores(res, 100, 0.5)
teacher.display()
```
