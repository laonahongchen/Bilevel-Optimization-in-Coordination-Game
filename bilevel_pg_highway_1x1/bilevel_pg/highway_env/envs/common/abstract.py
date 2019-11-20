from __future__ import division, print_function, absolute_import
import copy
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
from highway_env import utils
from highway_env.envs.common.observation import observation_factory
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.control import MDPVehicle
from highway_env.vehicle.dynamics import Obstacle
from bilevel_pg.bilevelpg.spaces import Discrete, Box, MASpace, MAEnvSpec


class AbstractEnv(gym.Env):
    """
        A generic environment for various tasks involving a vehicle driving on a road.

        The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
        velocity. The action space is fixed, but the observation space and reward function must be defined in the
        environment implementations.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    ACTIONS = {0: 'IDLE',
               1: 'LANE_LEFT',
               2: 'LANE_RIGHT',
               3: 'FASTER',
               4: 'SLOWER'}
    
    '''
    ACTIONS = {0: 'IDLE',
               1: 'FASTER',
              2: 'SLOWER'}
    '''
    """
        A mapping of action indexes to action labels
    """
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    """
        A mapping of action labels to action indexes
    """

    SIMULATION_FREQUENCY = 15
    """
        The frequency at which the system dynamics are simulated [Hz]
    """

    PERCEPTION_DISTANCE = 5.0 * MDPVehicle.SPEED_MAX
    """
        The maximum distance of any vehicle present in the observation [m]
    """

    DEFAULT_CONFIG = {
        "observation": {
            "type": "TimeToCollision"
        },
        "policy_frequency": 1,  # [Hz]
        "screen_width": 1200,
        "screen_height": 150
    }

    def __init__(self, config=None):
        # Configuration
        self.config = config
        if not self.config:
            self.config = self.DEFAULT_CONFIG.copy()

        # Seeding
        self.np_random = None
        self.seed()
        
        

        # Scene
        self.road = None
        self.vehicle = None

        

        # Spaces
        self.observation = None
        self.define_spaces()
        self.level_agent_num = 2
        self.action_num = 5
        self.num_state = 8
        #self.num_state = (2 * self.level_agent_num + 1) * 5

        self.agent_num = 2
        self.leader_num = 1
        self.follower_num = 1
        self.merge_start_x = 220
        self.merge_end_x = 310
        self.next_put_x = 500
        self.agents = []
        self.train_agents = []

        self.is_vehicles_valid = [False] * self.agent_num

        self.action_spaces = MASpace(tuple(Discrete(self.action_num) for _ in range(self.agent_num)))
        self.observation_spaces = MASpace(tuple(Discrete(self.num_state) for _ in range(self.agent_num)))
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)

        # Running
        self.time = 0
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False
        self.good_merge_flag = False
        self.episode_merge_record = []
        self.episodes_reward_0 = []
        self.episodes_reward_1 = []
        self.episode_target_merge_record = []
        self.sim_max_step = 5
        self.epsilon = 0.3

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config):
        if config:
            self.config.update(config)

    def define_spaces(self):
        # changing ---------------------------------------------
        #self.action_space = spaces.Discrete(len(self.ACTIONS)) 
        #self.action_spaces = [spaces.Discrete(len(self.ACTIONS)) for _ in range(len(self.all_vehicles))]
        #print("Action Spaces: ", self.action_spaces)

        if "observation" not in self.config:
            raise ValueError("The observation configuration must be defined")
        #self.observation = observation_factory(self, self.config["observation"])
        #self.observation_space = self.observation.space()
        self.observations = observation_factory(self, self.config["observation"])
        #self.observations_space = self.observations.space()
        #print("Observations Spaces", self.observation_space)

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError()

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        raise NotImplementedError()

    def _cost(self, actions):
        """
            A constraint metric, for budgeted MDP.

            If a constraint is defined, it must be used with an alternate reward that doesn't contain it
            as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        return None, self._reward(actions)

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        
        self.time = 0
        self.sim_step = 0
        self.episode_reward_0 = 0
        self.episode_reward_1 = 0
        #self.is_vehicles_valid = [False] * self.agent_num
        self.merge_count += 1
        #self.epsilon *= 0.998
        self.done = False
        self.define_spaces()
        return self.observations.get_observations()
        #return self.observation.observe()


    def check_vehicles_valid(self):
        #if self.road.vehicles[0].position[0] > self.road.vehicles[1].position[0]:
        #    self.good_merge_flag = True
        for i in range(len(self.road.vehicles)):
            if self.road.vehicles[i].position[0] > self.merge_start_x and self.road.vehicles[i].position[0] < self.merge_end_x:
                if not self.is_vehicles_valid[i]:
                    self.is_vehicles_valid[i] = True
                    #self.road.vehicles[i].velocity = np.random.uniform(30, 40)
            
            

    def step(self, actions):
        """
            Perform an action and step the environment dynamics.

            The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self._simulate(actions)

        obs = self.observations.get_observations()
        rewards = self._reward(actions)
        self.correct_merge_flag = False
        terminal = np.array([self._is_terminal()] * self.agent_num)
        if terminal[0]:
            if self.correct_merge_flag:
                self.correct_merge_count += 1
                self.episode_merge_record.append(1)
                if self.road.vehicles[0].position[0] > self.road.vehicles[1].position[0]:
                    rewards[0] += 50
                    rewards[1] += 10
                    self.episode_target_merge_record.append([1])
                else:
                    rewards[0] += 10
                    rewards[1] += 50
                    self.episode_target_merge_record.append([0])
            else:
                if self.road.vehicles[0].position[0] < 260:
                    rewards -= 25
                elif self.road.vehicles[0].position[0] < 265:
                    rewards -= 20
                elif self.road.vehicles[0].position[0] < 270:
                    rewards -= 15
                elif self.road.vehicles[0].position[0] < 275:
                    rewards -= 10
                else:
                    rewards -= 5
                self.episode_merge_record.append(0)
                self.episode_target_merge_record.append([0])
            
        self.episode_reward_0 += rewards[0]
        self.episode_reward_1 += rewards[1]
        self.sim_step += 1
        if terminal[0]:
            self.episodes_reward_0.append(self.episode_reward_0)
            self.episodes_reward_1.append(self.episode_reward_1)
        costs = self._cost(actions)
        info = {'cost': costs, "c_": costs}
        #self.check_vehicles_valid()

        return obs, rewards, terminal, info

    def _simulate(self, actions=None):
        """
            Perform several steps of simulation with constant action
        """
        
        for k in range(int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])):
            if actions is not None and \
                    self.time % int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"]) == 0:
                # Forward action to the vehicle
                #print(self.ACTIONS[action])
                #print("TEST length", len(self.all_vehicles))
                for i in range(len(self.road.vehicles)):
                    self.road.vehicles[i].act(self.ACTIONS[actions[i][0]])
                #self.vehicle.act(self.ACTIONS[action])

            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode
        
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            self.viewer.handle_events()
            return image
        elif mode == 'human':
            self.viewer.handle_events()
        self.should_update_rendering = False

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self):
        """
            Get the list of currently available actions.

            Lane changes are not available on the boundary of the road, and velocity changes are not available at
            maximal or minimal velocity.

        :return: the list of available actions
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_RIGHT'])
        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])
        return actions

    def _automatic_rendering(self):
        """
            Automatically render the intermediate frames while an action is still ongoing.
            This allows to render the whole video and not only single steps corresponding to agent decision-making.

            If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
            such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def simplify(self):
        """
            Return a simplified copy of the environment where distant vehicles have been removed from the road.

            This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, [-self.PERCEPTION_DISTANCE / 2, self.PERCEPTION_DISTANCE])

        return state_copy

    def change_vehicles(self, vehicle_class_path):
        """
            Change the type of all vehicles on the road
        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle and not isinstance(v, Obstacle):
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane=None):
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    raise NotImplementedError()
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def randomize_behaviour(self):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result
