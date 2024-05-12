import numpy as np
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from stable_baselines3 import PPO
from sensor_msgs.msg import LaserScan
from parameters import *
from pose_utils import get_yaw, PoseUtils
import os

model_file = f"/home/{os.getlogin()}/isaac_sim_ws/src/deep_learning_planner/rl_logs/runs/drl_policy_3/best_model.zip"


class TransformerDRLInference:
    def __init__(self):
        self.robot_frame = "base_link"
        self.global_plan = None
        self.goal = None
        self.laser_pool = []

        self.model = PPO.load(model_file)
        self._pose_utils = PoseUtils(robot_radius)
        self.laser_pool_capacity = interval * laser_length

        global_plan_topic_name = "/move_base/GlobalPlanner/robot_frame_plan"
        self._plan_sub = rospy.Subscriber(global_plan_topic_name, Path, self._path_callback, queue_size=1)
        self._goal_sub = rospy.Subscriber("/move_base/current_goal", PoseStamped, self._goal_callback, queue_size=1)
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10, latch=False)
        self._laser_sub = rospy.Subscriber("/scan", LaserScan, self._laser_callback, queue_size=1)
        rospy.Timer(rospy.Duration(secs=0, nsecs=20000000), self._cmd_inference)

    def _path_callback(self, msg):
        self.global_plan = msg

    def _goal_callback(self, msg):
        self.goal = msg

    def _laser_callback(self, laser_msg: LaserScan):
        if len(self.laser_pool) > self.laser_pool_capacity:
            self.laser_pool.pop(0)
        self.laser_pool.append(laser_msg.ranges)

    def _cmd_inference(self, event):
        if self.global_plan is None or self.goal is None or len(self.laser_pool) == 0:
            return
        observation = self._get_observation()
        if observation is None:
            return
        cmd_vel = Twist()
        action, _states = self.model.predict(observation)
        cmd_vel.linear.x = np.clip(action[0], min_vel_x, max_vel_x)
        cmd_vel.angular.z = np.clip(action[1], min_vel_z, max_vel_z)
        self._cmd_vel_pub.publish(cmd_vel)

    def _get_observation(self):
        pose = self._pose_utils.transform_pose(self.goal, self.robot_frame)
        if pose is None:
            return None
        assert isinstance(pose, PoseStamped)
        pos = np.array((pose.pose.position.x, pose.pose.position.y, get_yaw(pose.pose.orientation)))
        assert isinstance(self.global_plan, Path)
        global_plan = []
        for pose in self.global_plan.poses:
            assert isinstance(pose, PoseStamped)
            x = pose.pose.position.x
            y = pose.pose.position.y
            yaw = get_yaw(pose.pose.orientation)
            global_plan.append((x, y, yaw))
        global_plan = np.array(global_plan)
        if len(global_plan) > 0:
            global_plan = global_plan[:min(len(global_plan), look_ahead_poses * down_sample):down_sample, :]
            if len(global_plan) < look_ahead_poses:
                padding = np.stack([pos for _ in range(look_ahead_poses - len(global_plan))], axis=0)
                global_plan = np.concatenate([global_plan, padding], axis=0)
        else:
            global_plan = np.stack([pos for _ in range(look_ahead_poses)], axis=0)
        global_plan = global_plan / np.array([look_ahead_distance, look_ahead_distance, np.pi])
        pos = pos / np.array([look_ahead_distance, look_ahead_distance, np.pi])

        lasers = [self.laser_pool[-1] for _ in range(laser_length)]
        for i in range(laser_length - 1, 0, -1):
            prefix = len(self.laser_pool) - i * interval
            if prefix < 0:
                lasers[laser_length - i - 1] = self.laser_pool[0]
            else:
                lasers[laser_length - i - 1] = self.laser_pool[prefix]
        lasers = np.array(lasers) / laser_range
        return {
            "laser": lasers,
            "global_plan": global_plan,
            "goal": pos
        }


if __name__ == '__main__':
    rospy.init_node('transformer_drl_inference')
    inference = TransformerDRLInference()
    rospy.spin()
