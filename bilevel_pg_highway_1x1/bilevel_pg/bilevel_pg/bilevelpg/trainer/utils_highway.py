# Created by yingwen at 2019-06-30
from copy import deepcopy
import numpy as np
import tensorflow as tf

num_sample = 10

'''
checked yet
'''

def add_target_actions(env, sampler, batch_n, agents, train_agents, batch_size):
    target_actions_n = []
    
    '''
      if an invalid car has not merged yet, then its action is idle until
      reaching to the merge starting point; if an invalid car has already merged, then
      we will not train our model using experience from that car
    '''
    for i in range(sampler.leader_num):
        #mask = batch_n[i]["next_valid_conditions"].reshape(batch_size, )
        #print(mask, train_agents[sampler.leader_idx].act(batch_n[i]['next_observations'], use_target=True))
        target_actions_n.append(train_agents[0].act(batch_n[i]['next_observations'], use_target=True))
        #print(train_agents[sampler.leader_idx].act(batch_n[i]['next_observations'], use_target=True).shape)
       
    target_leader_actions = np.array(target_actions_n)
    #print(target_actions_n.shape)  # (leader_num, 32)
    #print(batch_n[i]['next_observations'].shape) # (32, 15)
    
    for i in range(sampler.leader_num, env.agent_num):
        #mask = batch_n[i]["next_valid_conditions"].reshape(batch_size, )
        #next_obs_v_idxes = batch_n[i]['next_obs_v_idxes']
        
        # ordered leader idx according to distance to leader vehicle
        
        #print(tf.one_hot(target_leader_actions[0, :], env.action_num))
        mix_obs = np.zeros((batch_size, env.num_state + env.action_num))
        mix_obs[:, :env.num_state] = batch_n[i]['next_observations']
        #print(batch_n[i]['next_observations'].shape)
        #print(next_obs_v_idxes[0])
        #print(target_leader_actions[0, :])
        #print(tf.one_hot(target_leader_actions[0, :], env.action_num))
        mix_obs[:, env.num_state:] = tf.one_hot(target_leader_actions[0, :], env.action_num)
        #print(target_leader_actions.shape)
        
        target_actions_n.append(train_agents[1].act(mix_obs, use_target=True))

    #print(np.array(target_actions_n).shape)
    for i in range(len(agents)):
        target_actions = np.array(target_actions_n[i])
        opponent_target_actions = np.reshape(np.delete(deepcopy(target_actions_n), i, 0), (batch_size, -1))
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