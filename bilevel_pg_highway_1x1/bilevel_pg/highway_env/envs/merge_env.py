from __future__ import division, print_function, absolute_import
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.dynamics import Obstacle


class MergeEnv(AbstractEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """

    COLLISION_REWARD = -1
    RIGHT_LANE_REWARD = 0.1
    HIGH_VELOCITY_REWARD = 5
    MERGING_VELOCITY_REWARD = -0.5
    LANE_CHANGE_REWARD = -0.05

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics_MA"
        },
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 850,
        "screen_height": 200,
        "centering_position": [0.3, 0.5]
    }

    def __init__(self):
        self.correct_merge_count = 0
        self.target_merge_count = 0
        self.merge_count = 0
        self.correct_merge_flag = False
        super(MergeEnv, self).__init__()
        self._make_road()
        #self.reset()




    def _reward(self, actions):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: 0,
                         1: self.LANE_CHANGE_REWARD,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        rewards = []
        '''
        reward_0 = self.COLLISION_REWARD * self.road.vehicles[0].crashed \
            + self.HIGH_VELOCITY_REWARD * self.road.vehicles[0].velocity_index / (self.road.vehicles[0].SPEED_COUNT - 1)
        
        if self.road.vehicles[0].lane_index == ("b", "c", 2):
            reward_0 += self.MERGING_VELOCITY_REWARD * \
                (self.road.vehicles[0].target_velocity - self.road.vehicles[0].velocity) / self.road.vehicles[0].target_velocity
        
        reward_0 = utils.remap(action_reward[actions[0][0]] + reward_0, [self.COLLISION_REWARD, self.HIGH_VELOCITY_REWARD],
                    [0,1])
        '''
        rewards.append(0)
        '''
        reward_1 = self.COLLISION_REWARD * self.road.vehicles[1].crashed \
            + self.HIGH_VELOCITY_REWARD * self.road.vehicles[1].velocity_index / (self.road.vehicles[1].SPEED_COUNT - 1)
        
        if self.road.vehicles[1].lane_index == ("b", "c", 2):
            reward_1 += self.MERGING_VELOCITY_REWARD * \
                        (self.road.vehicles[1].target_velocity - self.road.vehicles[1].velocity) / self.road.vehicles[1].target_velocity
        reward_1 = utils.remap(action_reward[actions[1][0]] + reward_1, [self.COLLISION_REWARD, self.HIGH_VELOCITY_REWARD],
                    [0,1])
        '''
        rewards.append(0)
        
        return np.array(rewards)
        #reward = self.COLLISION_REWARD * self.vehicle.crashed \
         #        + self.RIGHT_LANE_REWARD * self.vehicle.lane_index[2] / 1 \
          #       + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        #for vehicle in self.road.vehicles:
        #    if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
        #        reward += self.MERGING_VELOCITY_REWARD * \
        #                  (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity

        #return utils.remap(action_reward[action] + reward,
                      #     [self.COLLISION_REWARD, self.HIGH_VELOCITY_REWARD + self.RIGHT_LANE_REWARD],
                        #   [0, 1])

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        pass_flag = False
        for vehicle in self.road.vehicles:
            if vehicle.crashed:
                #print("Crashed!")
                #self.is_vehicles_valid = [False] * self.agent_num
                return True
            if vehicle.position[0] > self.merge_end_x:
                pass_flag = True
        if pass_flag: 
            self.correct_merge_flag = True
            #print("Correct merge! Correct merge count = ", self.correct_merge_count)
            #print("Merge count = ", self.merge_count)
            #self.is_vehicles_valid = [False] * self.agent_num
            return True
        else:
            return False
        #return self.vehicle.crashed or self.vehicle.position[0] > 370

    def reset(self):
        #self._make_road()
        self._make_vehicles()
        return super(MergeEnv, self).reset()

    def _make_road(self):
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [200, 50, 30, 800]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [c, c]]
        line_type_merge = [[c, s], [c, s]]
        '''
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))
        '''
        i = 1
        net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
        net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
        net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))
        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        
        
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random)
        #road.vehicles.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        main_init_xpos = 215
        auxi_init_xpos = 215
        init_interval = 20
        road.vehicles = []
        for i in range(self.leader_num):
            velocity = np.random.uniform(38, 40)
            road.vehicles.append(MDPVehicle(road, road.network.get_lane(("a", "b", 0)).position(main_init_xpos, 0), index = i, velocity=velocity))
            main_init_xpos += init_interval

        for i in range(self.follower_num):
            velocity = np.random.uniform(38, 40)
            road.vehicles.append(MDPVehicle(road, road.network.get_lane(("j", "k", 0)).position(auxi_init_xpos, 0), index = i + self.leader_num, velocity=velocity))
            auxi_init_xpos += init_interval

        
        self.vehicle = road.vehicles[0]
        road.leader_num = self.leader_num
        road.follower_num = self.follower_num
        

register(
    id='merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)
