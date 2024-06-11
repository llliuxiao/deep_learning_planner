# utils
import numpy as np
import pandas as pd
import argparse
import os

# ros
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
import tf2_ros
from tf2_ros import Buffer
from tf2_ros.transform_listener import TransformListener

# customer module
from parameters import *
from pose_utils import calculate_geodesic_distance


class Evaluate:
    def __init__(self, out_file, planner, robot_frame, goal_topic_name,
                 imu_topic_name, odom_topic_name, global_plan_name):
        self.robot_frame = robot_frame
        self.out_file = out_file
        self.planner = planner

        self.goal = None
        self.imu = None
        self.odom = None
        self.digress_distance = None
        self.goal_reached = True
        self.start_time = None
        self.geodesic_distance = None

        self.linear_acc_pool = []
        self.angular_pool = []
        self.digress_distance_pool = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.goal_sub = rospy.Subscriber(goal_topic_name, PoseStamped, self.goal_callback, queue_size=1)
        self.imu_sub = rospy.Subscriber(imu_topic_name, Imu, self.imu_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic_name, Odometry, self.odom_callback, queue_size=1)
        self.global_plan_sub = rospy.Subscriber(global_plan_name, Path, self.path_callback, queue_size=1)

        if not os.path.exists(out_file):
            self.df = pd.DataFrame(columns=["std_acc", "avg_acc", "max_acc", "avg_angular", "avg_digress",
                                            "max_digress", "geodesic_dis", "run_time", "algorithm"])
        else:
            self.df = pd.read_csv(out_file)
        rospy.Timer(rospy.Duration(secs=0, nsecs=40000000), self.evaluate)

    def get_distance_to_goal(self) -> float:
        try:
            robot_pose = self.tf_buffer.lookup_transform("map", self.robot_frame, rospy.Time(0))
            assert isinstance(robot_pose, TransformStamped)
        except tf2_ros.TransformException:
            rospy.logfatal("could not get robot pose")
            return math.inf
        assert isinstance(self.goal, PoseStamped)
        delta_x = self.goal.pose.position.x - robot_pose.transform.translation.x
        delta_y = self.goal.pose.position.y - robot_pose.transform.translation.y
        return math.sqrt(delta_x ** 2 + delta_y ** 2)

    def evaluate(self, event):
        if self.goal_reached or self.goal is None or self.imu is None or self.odom is None or self.digress_distance is None:
            return
        assert isinstance(self.imu, Imu)
        assert isinstance(self.odom, Odometry)
        self.linear_acc_pool.append(self.imu.linear_acceleration.x)
        self.angular_pool.append(abs(self.odom.twist.twist.angular.z))
        self.digress_distance_pool.append(self.digress_distance)
        distance = self.get_distance_to_goal()
        if distance <= goal_radius:
            self.goal_reached = True
            self.goal = None
            self.df.loc[len(self.df)] = [
                np.std(self.linear_acc_pool),
                np.mean(self.linear_acc_pool),
                np.max(self.linear_acc_pool),
                np.mean(self.angular_pool),
                np.mean(self.digress_distance_pool),
                np.max(self.digress_distance_pool),
                self.geodesic_distance,
                (rospy.Time.now() - self.start_time).to_sec(),
                self.planner
            ]
            self.df.to_csv("evaluation.csv", index=False)
            rospy.loginfo("reach the goal")
            self.linear_acc_pool.clear()
            self.angular_pool.clear()
            self.digress_distance_pool.clear()
            self.goal = None
        else:
            rospy.logfatal(f"not reach, {distance} left")

    def imu_callback(self, msg):
        self.imu = msg

    def odom_callback(self, msg: Odometry):
        self.odom = msg
        if self.start_time is None and (msg.twist.twist.linear.x > 0.05 or abs(msg.twist.twist.angular.z) > 0.05):
            self.start_time = rospy.Time.now()

    def goal_callback(self, msg):
        if self.goal is None:
            self.goal_reached = False
            self.start_time = None
            self.geodesic_distance = None
            rospy.logfatal("receive new goal")
            self.goal = msg

    def path_callback(self, path: Path):
        poses = []
        for pose in path.poses:
            assert isinstance(pose, PoseStamped)
            poses.append((pose.pose.position.x, pose.pose.position.y))
        poses = np.array(poses)
        if len(poses) > 0:
            self.digress_distance = np.min(np.linalg.norm(poses, axis=1))
        if self.geodesic_distance is None:
            self.geodesic_distance = calculate_geodesic_distance(poses)


if __name__ == "__main__":
    rospy.init_node("evaluate")
    parse = argparse.ArgumentParser(description="script to evaluate motion planner algorithm")
    parse.add_argument("-r", "--robot", type=str, default="sim", help="sim or agv234")
    parse.add_argument("-t", "--type", type=str, default="imitation", help="the evaluated algorithm")
    parse.add_argument("-o", "--output", type=str, default="out.csv", help="the file to save the result")
    args = parse.parse_args()
    if args.robot == "sim":
        eval_kwargs = dict(
            out_file=args.output,
            planner=args.type,
            robot_frame="base_link",
            goal_topic_name="/move_base/current_goal",
            imu_topic_name="/imu",
            odom_topic_name="/odom",
            global_plan_name="/move_base/GlobalPlanner/robot_frame_plan"
        )
    else:
        eval_kwargs = dict(
            out_file=args.output,
            planner=args.type,
            robot_frame="base_footprint",
            goal_topic_name="/robot4/move_base/current_goal",
            imu_topic_name="/robot4/imu_data",
            odom_topic_name="/vm_gsd601/gr_canopen_vm_motor/mobile_base_controller/odom",
            global_plan_name="/robot4/move_base/GlobalPlanner/robot_frame_plan",
        )
    evaluation = Evaluate(**eval_kwargs)
    rospy.spin()
