# Lenient and Optimistic MA-DRL Approaches

This framework provides implementations for the algorithms and environments discussed in:

<b>Negative Update Intervals in Deep Multi-Agent Reinforcement Learning</b></br>
Gregory Palmer, Rahul Savani, Karl Tuyls. </br>
Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS) </br>
[&nbsp;<a href="https://arxiv.org/abs/1809.05096" target="_blank">arXiv</a>&nbsp;]</br>
</br>
<b>Lenient Multi-Agent Deep Reinforcement Learning</b></br>
Gregory Palmer, Karl Tuyls, Daan Bloembergen, Rahul Savani. </br>
Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS) </br>
[&nbsp;<a href="https://dl.acm.org/citation.cfm?id=3237451" target="_blank">ACM Digital Library</a>&nbsp;|&nbsp; <a href="https://arxiv.org/abs/1707.04402v2" target="_blank">arXiv</a>&nbsp;]

## Apprentice Firemen Game

### City Grid Layouts 
<table>
<tr>
<td> 
<figure>
<img src='img/afg_env1.png' height='350'/><br>
<font size="1"><figcaption>Layout 1: Observable Irrevocable Decision</figcaption></font>
</figure>
</td>
<td> 
<figure>
<img src='img/afg_env2.png' height='350'/><br>
<font size="1"><figcaption>Layout 2: Irrevocable Decisions in Seclusion</figcaption></font>
</figure>
</td>
</tr>
</table>

### Access Points Layouts

![Access Point Layouts](img/Access_Points.png "Access Point Layouts")

### Rewards

Reward structures for Deterministic (DET), Partially Stochastic (PS) and Fully Stochastic (FS) Apprentice Firemen Games, to be interpreted as rewards for (Agent 1, Agent 2). For (B,B) within PS and FS 1.0 is yielded on 60% of occasions. Rewards are sparse, received only at the end of an episode after the fire has been extinguished.

<img src='img/reward_tables.png' alt="Reward Tables" width="100%"/>

### Running the AFG

The environment flag can be used to specify the layout (V{1,2}), number of civilians (C{INT}) and the reward structure ({DET,PS,FS}):

	python apprentice_firemen.py --environment ApprenticeFiremen_V{1,2}_C{INT}_{DET,PS,FS}

For Layout 3 the number of access points must be specified:

	python apprentice_firemen.py --environment ApprenticeFiremen_V{1,2}_C{INT}_{DET,PS,FS}_{1,2,3,4}AP

Further layout config files can be added under:

	./env/ApprenticeFiremen/

## Fully Cooperative Multi-agent Object Transporation Problem (CMOTP)

### Layouts:

1. CMOTP_V1 = Original
2. CMOTP_V2 = Narrow Passages
3. CMOTP_V3 = Stochastic Reward

![Original](img/v1.png "Original ")
![Narrow Passages](img/v2.png "Narrow Passages")
![Stochastic Reward](img/v3.png "Stochastic Reward")


Further layout config files can be added under:

	./env/cmotp/

Simply copy one of the existing envconfig files and make your own amendments.

### Running the CMOTP:

Agents can be trained via: 

	python cmotp.py --environment CMOTP_V1 --render False

## Choosing a MA-DRL algorithm

For lenient/hysteretic/nui agents:   

	python {apprentice_firemen, cmotp}.py --madrl {hysteretic, leniency, nui}

## Hyperparameters

Hyperparameters can be adjusted in:

	./config.py

## Results

Results from individual runs can be found under:

	./Results/

## Using your own agents

To run the above domains using your own agents modify the following files in 

	./main.py

```python
# !---Import your agent class here ---! 
from agent import Agent
# !---Import your agent class here ---! 
```

```python
# !--- Store agent instances in agents list ---!
# Example:
agents = []
config = Config(env.dim, env.out, madrl=FLAGS.madrl, gpu=FLAGS.processor)
# Agents are instantiated
for i in range(FLAGS.agents): 
    agent_config = deepcopy(config)
    agents.append(Agent(agent_config)) # Init agent instances
# !--- Store agent instances in agents list ---! 
```

```python
# !--- Get actions from each agent ---!
actions.append(agent.move(observation)) 
# !--- Get actions from each agent ---!
```

```python
# !--- Pass feedback to each agent ---!
for agent, o, r in zip(agents, observations, rewards):
    agent.feedback(r, t, o) 
# !--- Pass feedback to each agent ---!
```

