import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry

import numpy as np
import math

import tf2_ros
from tf2_ros import Buffer
from tf2_ros.transform_listener import TransformListener

robot_frame = "base_footprint"
goal_topic_name = "/robot4/move_base/current_goal"
imu_topic_name = "/robot4/imu_data"
odom_topic_name = "/vm_gsd601/gr_canopen_vm_motor/mobile_base_controller/odom"

goal_tolerance = 0.3


class Evaluate:
    def __init__(self):
        self.goal = None
        self.imu = None
        self.odom = None
        self.goal_reached = True
        self.start_time = rospy.Time.now()
        self.linear_acc_pool = []
        self.angular_pool = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.goal_sub = rospy.Subscriber(goal_topic_name, PoseStamped, self.goal_callback, queue_size=1)
        self.imu_sub = rospy.Subscriber(imu_topic_name, Imu, self.imu_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic_name, Odometry, self.odom_callback, queue_size=1)

        rospy.Timer(rospy.Duration(secs=0, nsecs=40000000), self.evaluate)

    def get_distance_to_goal(self) -> float:
        try:
            robot_pose = self.tf_buffer.lookup_transform("map", robot_frame, rospy.Time(0))
            assert isinstance(robot_pose, TransformStamped)
        except tf2_ros.TransformException:
            rospy.logfatal("could not get robot pose")
            return math.inf
        assert isinstance(self.goal, PoseStamped)
        delta_x = self.goal.pose.position.x - robot_pose.transform.translation.x
        delta_y = self.goal.pose.position.y - robot_pose.transform.translation.y
        return math.sqrt(delta_x ** 2 + delta_y ** 2)

    def evaluate(self, event):
        if self.goal_reached or self.goal is None or self.imu is None or self.odom is None:
            return
        assert isinstance(self.imu, Imu)
        assert isinstance(self.odom, Odometry)
        self.linear_acc_pool.append(self.imu.linear_acceleration.x)
        self.angular_pool.append(abs(self.odom.twist.twist.angular.z))
        distance = self.get_distance_to_goal()
        if distance <= goal_tolerance:
            self.goal_reached = True
            self.goal = None
            rospy.logfatal(f"the deviation of linear acceleration is {np.std(np.array(self.linear_acc_pool))}")
            rospy.logfatal(f"the avg of the angular velocity is {np.mean(np.array(self.angular_pool))}")
            rospy.logfatal(f"running time is {(rospy.Time.now() - self.start_time).to_sec()}")
            rospy.logfatal("reach the goal")
            self.linear_acc_pool.clear()
            self.angular_pool.clear()

    def imu_callback(self, msg):
        self.imu = msg

    def odom_callback(self, msg):
        self.odom = msg

    def goal_callback(self, msg):
        if self.goal is None:
            self.goal_reached = False
            self.start_time = rospy.Time.now()
            rospy.logfatal("receive new goal")
        else:
            distance = math.sqrt((msg.pose.position.x - self.goal.pose.position.x) ** 2 +
                                 (msg.pose.position.y - self.goal.pose.position.y) ** 2)
            if distance > 0.05:
                self.goal_reached = False
                self.start_time = rospy.Time.now()
                rospy.logfatal("receive new goal")
        self.goal = msg


if __name__ == "__main__":
    rospy.init_node("evaluate")
    eval = Evaluate()
    rospy.spin()
