class Pose_t:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.heading = 0
        self.arc_len = 0
        self.curvature = 0

class Path_t:
    def __init__(self):
        self.path = []


class Obstacle:
    def __init__(self):
        self.pose = Pose_t()
        self.length = 0
        self.width = 0
