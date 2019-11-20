# Created by yingwen at 2019-06-30
from copy import deepcopy
import numpy as np
import tensorflow as tf

num_sample = 10


def add_target_actions(env, sampler, batch_n, agents, train_agents, batch_size):
    target_actions_n = []
    # batch (2, )
    # leader action
    #target_actions_n.append(agents[sampler.leader_idx].act(batch_n[sampler.leader_idx]['next_observations'], use_target=True))
    #print(batch_n[0]['next_observations'].shape) (32, 25)
    for i in range(sampler.leader_num):
        if env.is_vehicles_valid[i]:
            target_actions_n.append(train_agents[sampler.leader_idx].act(batch_n[i]['next_observations'], train_agents[sampler.follower_idx]))
            #print(train_agents[sampler.leader_idx].act(batch_n[i]['next_observations'], use_target=True).shape)
        else:
            idle_actions = [1] * batch_size
            #print(train_agents[sampler.leader_idx].act(batch_n[i]['next_observations'], use_target=True).shape)
            target_actions_n.append(np.array(idle_actions)) # idle action

    #print(target_actions_n.shape)  # (leader_num, 32)
    #print(batch_n[i]['next_observations'].shape) # (32, 15)
    
    for i in range(sampler.leader_num, env.agent_num):
        #env.is_vehicles_valid[i] = True
        if env.is_vehicles_valid[i]:
            v = env.road.vehicles[i]
            closest_leaders = env.road.closest_leader_vehicles_to(v)
            # ordered leader idx according to distance to leader vehicle
            leader_actions_concat = np.zeros((batch_size, sampler.level_agent_num * env.action_num))
            
            for j in range(len(closest_leaders)):
                #print(target_actions_n[closest_leaders[i].index])
                arr = np.array(tf.one_hot(target_actions_n[closest_leaders[j].index], env.action_num))
                #print(arr)   # (32, 5)
                leader_actions_concat[:, 5*j:5*(j+1)] = arr
                #print(leader_actions_concat)
                #leader_actions_concat = np.hstack(leader_actions_concat, tf.one_hot(target_actions_n[:, leader.index], env.action_num))
            
            
            mix_obs = np.zeros((batch_size, env.num_state + env.action_num * sampler.level_agent_num))
            mix_obs[:, :env.num_state] = batch_n[i]['next_observations']
            mix_obs[:, env.num_state:] = leader_actions_concat
        
            target_actions_n.append(train_agents[sampler.follower_idx].act(mix_obs, use_target=True))
            #print(train_agents[sampler.follower_idx].act(mix_obs, use_target=True).shape)
        else: 
            idle_actions = [1] * batch_size
            target_actions_n.append(np.array(idle_actions))  #idle action
    #print(np.array(target_actions_n).shape)
    for i in range(len(agents)):
        target_actions = np.array(target_actions_n[i])
        #print(target_actions, target_actions.shape)
        # long = target_actions.shape[0]
        # target_actions.reshape(-1, 1)
        # print(target_actions.shape)
        #print(np.array(target_actions_n).shape)
        opponent_target_actions = np.reshape(np.delete(deepcopy(target_actions_n), i, 0), (batch_size, -1))

        # print(opponent_target_actions.shape)

        target_actions = np.concatenate((target_actions.reshape(-1, 1), opponent_target_actions), 1)
        assert target_actions.shape[0] == batch_size
        batch_n[i]['target_actions'] = target_actions
    return batch_n
'''

def add_target_actions(batch_n, agents, batch_size):

    # the first agent in agents should be the leader agent while the second should be the follower agent


    target_actions_n = []
    # for i, agent in enumerate(agents):
    #     print(batch_n[i]['next_observations'].shape)
    #     target_actions_n.append(agent.act(batch_n[i]['next_observations'], use_target=True))

    sample_follower = []
    for i in range(num_sample):
        sample_follower.append(agents[1].act())

    target_actions_n.append(agents[0].act(tf.concat(batch_n[0]['next_observations'], sample_follower), use_target = True))

    for i in range(len(agents)):
        target_actions = target_actions_n[i]
        opponent_target_actions = np.reshape(np.delete(deepcopy(target_actions_n), i, 0), (batch_size, -1))
        target_actions = np.concatenate((target_actions, opponent_target_actions), 1)
        assert target_actions.shape[0] == batch_size
        batch_n[i]['target_actions'] = target_actions
    return batch_n
'''

def add_recent_batches(batches, agents, batch_size):
    for batch, agent in zip(batches, agents):
        recent_batch = agent.replay_buffer.recent_batch(batch_size)
        batch['recent_observations'] = recent_batch['observations']
        batch['recent_actions'] = recent_batch['actions']
        batch['recent_opponent_actions'] = recent_batch['opponent_actions']
    return batches


def add_annealing(batches, step, annealing_scale=1.):
    annealing = .1 + np.exp(-0.1*max(step-10, 0)) * 500
    annealing = annealing_scale * annealing
    for batch in batches:
        batch['annealing'] = annealing
    return batches


def get_batches(agents, batch_size):
    assert len(agents) > 1
    batches = []
    indices = agents[0].replay_buffer.random_indices(batch_size)
    for agent in agents:
        batch = agent.replay_buffer.batch_by_indices(indices)
        batches.append(batch)
    return batches


get_extra_experiences = {
    'annealing': add_annealing,
    'recent_experiences': add_recent_batches,
    'target_actions': add_target_actions,
}