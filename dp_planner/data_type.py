import math
class Pose_t:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.time = 0
        self.heading = 0
        self.arc_len = 0
        self.curvature = 0

class FrenetPose_t:
    def __init__(self, s = 0, l = 0, t = 0):
        self.s = s
        self.l = l
        self.t = t
        
    def __key(self):
        return (self.s, self.l, self.t)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.__key() == other.__key()

class Path_t:
    def __init__(self):
        self.path = []

class Obstacle:
    def __init__(self):
        self.pose = Pose_t()
        self.length = 0
        self.width = 0
        self.radius = 0
        self.time = 0
        self.moving = True
        self.predict_pose = []

def dist(pose_1, pose_2):
    return math.hypot(pose_1.x - pose_2.x, pose_1.y - pose_2.y)
