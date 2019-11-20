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
    HIGH_VELOCITY_REWARD = 0.2
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
        super(MergeEnv, self).__init__()
        
        self.reset()




    def _reward(self, actions):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        rewards = []
        for i in range(len(self.all_vehicles)):
            reward = self.COLLISION_REWARD * self.all_vehicles[i].crashed \
                + self.HIGH_VELOCITY_REWARD * self.all_vehicles[i].velocity_index / (self.all_vehicles[i].SPEED_COUNT - 1)
            
            if self.all_vehicles[i].lane_index == ("b", "c", 2) and isinstance(self.all_vehicles[i], ControlledVehicle):
                reward += self.MERGING_VELOCITY_REWARD * \
                            (self.all_vehicles[i].target_velocity - self.all_vehicles[i].velocity) / self.all_vehicles[i].target_velocity
            reward = utils.remap(action_reward[actions[i][0]] + reward, [self.COLLISION_REWARD, self.HIGH_VELOCITY_REWARD + self.RIGHT_LANE_REWARD],
                        [0,1])
            rewards.append(reward)
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
        for vehicle in self.road.vehicles:
            if vehicle.crashed or self.vehicle.position[0] > 370:
                return True
        #return self.vehicle.crashed or self.vehicle.position[0] > 370

    def reset(self):
        self.all_vehicles = []
        self._make_road()
        self._make_vehicles()
        return super(MergeEnv, self).reset()

    def _make_road(self):
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
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
        road.vehicles.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        leader_v_one = MDPVehicle(road, road.network.get_lane(("a", "b", 0)).position(100, 0), velocity=30)
        road.vehicles.append(leader_v_one)
        self.all_vehicles.append(leader_v_one)  
        
        leader_v_two = MDPVehicle(road, road.network.get_lane(("a", "b", 0)).position(80, 0), velocity=30)
        road.vehicles.append(leader_v_two)
        self.all_vehicles.append(leader_v_two)  

        #test_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(50, 0), velocity=30)
        #road.vehicles.append(test_vehicle)
        #self.all_vehicles.append(test_vehicle)
        
        #other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        #road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), velocity=29))
        #road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), velocity=31))
        #road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), velocity=31.5))

        #merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), velocity=20)
        merging_v_one = MDPVehicle(road, road.network.get_lane(("j", "k", 0)).position(110, 0), velocity=30)
        #merging_v_one.target_velocity = 30
        road.vehicles.append(merging_v_one)
        
        self.vehicle = leader_v_one

        merging_v_two = MDPVehicle(road, road.network.get_lane(("j", "k", 0)).position(90, 0), velocity=30)
        #merging_v_two.target_velocity = 30
        road.vehicles.append(merging_v_two)

        self.all_vehicles.append(merging_v_one)
        self.all_vehicles.append(merging_v_two)

register(
    id='merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)
