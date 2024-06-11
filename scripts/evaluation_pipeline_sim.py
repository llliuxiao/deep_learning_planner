# utils
import sys
import os
import subprocess
import argparse

from nav_msgs.msg import Odometry

sys.path.append(f"/home/{os.getlogin()}/isaac_sim_ws/devel/lib/python3/dist-packages")
isaac_sim_python = f"/home/{os.getlogin()}/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh"

environment_python = f"/home/{os.getlogin()}/isaac_sim_ws/src/isaac_sim/scripts/environment.py"

import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from actionlib import SimpleActionClient
from isaac_sim.msg import ResetPosesAction, ResetPosesGoal, ResetPosesResult

import math

points = ((-10.0, 8.0, math.pi), (-15.0, 28.0, math.pi / 2))


class GoalPublisher:
    def __init__(self):
        self.goal_pub = SimpleActionClient("/move_base", MoveBaseAction)
        self.init_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        self.reset_client = SimpleActionClient("/reset", ResetPosesAction)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.isaac_flag = False

    def reset(self, x, y, yaw):
        for _ in range(10):
            self.cmd_vel_pub.publish(Twist())
            rospy.sleep(0.05)

        init_pose = PoseStamped()
        init_pose.header.frame_id = "map"
        init_pose.pose.position.x = x
        init_pose.pose.position.y = y
        init_pose.pose.orientation.w = math.cos(0.5 * yaw)
        init_pose.pose.orientation.z = math.sin(0.5 * yaw)

        isaac_init_poses = ResetPosesGoal()
        isaac_init_poses.poses.append(init_pose)
        isaac_init_poses.prefix.append(0)

        rospy.logfatal("reset in isaac sim")
        self.reset_client.wait_for_server()
        self.reset_client.send_goal_and_wait(isaac_init_poses)
        rospy.sleep(3.0)

        rospy.logfatal("reset amcl")
        amcl_init_pose = PoseWithCovarianceStamped()
        amcl_init_pose.header = init_pose.header
        amcl_init_pose.header.stamp = rospy.Time.now()
        amcl_init_pose.pose.pose = init_pose.pose
        self.init_pub.publish(amcl_init_pose)
        rospy.sleep(3.0)

    def send_goal(self, x, y, yaw):
        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.frame_id = "map"
        move_base_goal.target_pose.header.stamp = rospy.Time.now()
        move_base_goal.target_pose.pose.position.x = x
        move_base_goal.target_pose.pose.position.y = y
        move_base_goal.target_pose.pose.orientation.w = math.cos(0.5 * yaw)
        move_base_goal.target_pose.pose.orientation.z = math.sin(0.5 * yaw)
        rospy.logfatal("send goal")
        self.goal_pub.send_goal_and_wait(move_base_goal)

    def execute(self):
        self.reset(points[0][0], points[0][1], points[0][2])
        self.send_goal(points[1][0], points[1][1], points[1][2])

    def wait_for_isaac(self):
        while not self.isaac_flag:
            rospy.sleep(0.5)

    def odom_callback(self, msg: Odometry):
        self.isaac_flag = True


if __name__ == "__main__":
    rospy.init_node("evaluation_pipeline")
    parser = argparse.ArgumentParser(description="A evaluation pipeline for robot navigation")
    parser.add_argument("-s", "--scene", type=str, default="office", help="evaluation scene")
    args = parser.parse_args()
    scene = args.scene
    goal_pub = GoalPublisher()
    ##################################################################################################
    # start navigation
    navigation_process = subprocess.Popen(
        ["roslaunch", "isaac_sim", "single_robot_navigation.launch", f"map:={scene}"]
    )
    ##################################################################################################
    # start evaluation script
    evaluation_process = subprocess.Popen(
        ["python", "evaluation.py", "-r sim"]
    )
    goal_pub.wait_for_isaac()
    goal_pub.execute()

    navigation_process.terminate()
    navigation_process.wait()
    evaluation_process.terminate()
    evaluation_process.wait()
