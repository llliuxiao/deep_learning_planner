import unittest
import math
from pose_utils import calculate_robot_position, calculate_relevant_position


class MyTestCase(unittest.TestCase):
    def test_calculate_robot_pose_function(self):
        robot_pose = (2.5, 6.8, math.pi / 4)
        target_pose = (3.2, -3.5, math.pi / 6)
        relevance = calculate_relevant_position(robot_pose, target_pose)
        robot = calculate_robot_position(target_pose, relevance)
        self.assertAlmostEqual(robot[0], robot_pose[0], places=3)
        self.assertAlmostEqual(robot[1], robot_pose[1], places=3)
        self.assertAlmostEqual(robot[2], robot_pose[2], places=3)


if __name__ == '__main__':
    unittest.main()
