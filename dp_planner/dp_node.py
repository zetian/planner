# import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../math/")
from polynomial_curve1d import *
from data_type import *

class PathCost:
    def __init__(self):
        self.safety_cost = 0
        self.smoothness_cost = 0
        self.has_collision = False
        self.out_of_boundary = False

    def __add__(self, other):
        res = PathCost()
        res.safety_cost = self.safety_cost + other.safety_cost
        res.smoothness_cost = self.smoothness_cost + other.smoothness_cost
        res.has_collision = self.has_collision or other.has_collision
        res.out_of_boundary = self.out_of_boundary or other.out_of_boundary
        return res
    
    def __gt__(self, other):
        if self.safety_cost + self.smoothness_cost > other.safety_cost + other.smoothness_cost:
            return True
        else:
            return False
    def __lt__(self, other):
        if self.safety_cost + self.smoothness_cost < other.safety_cost + other.smoothness_cost:
            return True
        else:
            return False
    def __eq__(self, other):
        if self.safety_cost + self.smoothness_cost == other.safety_cost + other.smoothness_cost:
            return True
        else:
            return False


class DPNode:
    def __init__(self, s = 0, l = 0, t = 0):
        self.s = s
        self.l = l
        self.t = t
        self.min_cost = PathCost()
        self.parent_pose = FrenetPose_t()


class DPSearch:
    def __init__(self):
        self.planning_horizon = 100
        self.planning_interval = 10
        self.lateral_dist = 3.0
        self.lateral_interval = 0.5
        self.time_interval = 1
        self.time_horizon = 10
        self.obstacle_list = []
        self.reference_path = Path_t()
        self.search_points = [[[]]]
        self.search_map = dict()

    def CollisionCheck(self, pose):
        for obs in self.obstacle_list:
            if obs.time == pose.time:
                if dist(obs.pose, pose) < obs.radius:
                    return True
        return False

    def Initialize(self):
        init_node = DPNode(0, 0, 0)
        self.search_map[FrenetPose_t(0, 0, 0)] = init_node

        longitudinal = np.linspace(self.planning_interval, self.planning_horizon, self.planning_horizon/self.planning_interval)
        lateral = np.linspace(-self.lateral_dist, self.lateral_dist, self.lateral_dist*2/self.lateral_interval + 1)
        time = np.linspace(0, self.time_horizon, self.time_horizon/self.time_interval + 1)
        for s in longitudinal:
            for t in time:
                for d in lateral:
                    self.search_map[FrenetPose_t(s, d, t)] = DPNode(s, d, t)
                    print(s, ", ", d, ", ", t)

        
    # def UpdateCost(self, pre_node, cur_node, poly_curve):


    # def SetWaypoints(self):


    # def GenerateMinCostPath(self):
dp_search = DPSearch()
dp_search.Initialize()
# pt = FrenetPose_t()
# pt.s = 13
# pt2 = FrenetPose_t(1, 2, 3)
# print(pt.s)
# c1 = PathCost()
# c2 = PathCost()
# # c3 = PathCost()
# c1.safety_cost = 1
# c2.safety_cost = 22
# c3 = c1 + c2
# print(c3.safety_cost)


# Test QuarticPolynomialCurve1d
# start = [0, 10, 0]
# end = [50, 0, 0]
# time_end = 8
# quartic_poly = QuinticPolynomialCurve1d(start, end, time_end)
# time = np.linspace(0, time_end, 200)
# pos = quartic_poly.Evaluate(0, time)
# vel = quartic_poly.Evaluate(1, time)
# accel = quartic_poly.Evaluate(2, time)

# plt.figure()
# plt.subplot(311)
# plt.plot(time, pos, color = 'black')
# plt.title('Quartic Polynomial Curve')
# plt.ylabel('Position(m)')
# plt.xlim(0, time_end) 
# plt.subplot(312)
# plt.plot(time, vel, color = 'red')
# plt.ylabel('Velocity(m/s)')
# plt.xlim(0, time_end) 
# plt.subplot(313)
# plt.plot(time, accel)
# plt.xlabel('time(s)')
# plt.ylabel('Acceleration(m/s^2)')
# plt.xlim(0, time_end) 
# plt.show()

