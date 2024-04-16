from transformer_planner import TransformerPlanner
from parameters import *

import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import torch

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib


class TransformerPlanner006(TransformerPlanner):
    def __init__(self,
                 velocity_factor=0.6,
                 robot_frame="base_footprint",
                 scan_topic_name="/map_scan",
                 global_topic_name="/robot6/move_base/GlobalPlanner/robot_frame_plan",
                 goal_topic_name="/robot6/move_base/current_goal",
                 cmd_topic_name="/vm_gsd601/gr_canopen_vm_motor/mobile_base_controller/cmd_vel",
                 short_distance_movement_action="/robot6/short_distance_movement"):
        super().__init__(velocity_factor, robot_frame, scan_topic_name,
                         global_topic_name, goal_topic_name, cmd_topic_name)
        self.client = actionlib.SimpleActionClient(short_distance_movement_action, MoveBaseAction)
        self.client.wait_for_server(rospy.Duration(secs=10))

    def cmd_inference(self, event):
        if self.goal_reached or self.goal is None:
            return
        cmd_vel = Twist()
        self.wait_for_raw_data()
        tensor = self.make_tensor()
        if tensor is None:
            return
        else:
            laser_tensor, global_plan_tensor, goal_tensor, laser_mask = tensor
        self.model.train(False)
        with torch.no_grad():
            predict = self.model(laser_tensor, global_plan_tensor, goal_tensor, laser_mask)
        predict = torch.squeeze(predict)
        cmd_vel.linear.x = self.velocity_factor * predict[0].item()
        cmd_vel.angular.z = self.velocity_factor * predict[1].item()
        distance = self.get_distance_to_goal()
        assert isinstance(self.global_plan, Path)
        if distance <= goal_radius and len(self.global_plan.poses) < 10:
            self.goal_reached = True
            self.goal = None
            self.cmd_vel_pub.publish(Twist())
            self.position_mode()
            rospy.logfatal("reach the goal")
        elif distance <= deceleration_tolerance and len(self.global_plan.poses) < 10:
            linear = self.linear_deceleration(1.0 * self.velocity_factor,
                                              deceleration_tolerance - distance,
                                              deceleration_tolerance - goal_radius)
            cmd_vel.angular.z = linear / cmd_vel.linear.x * cmd_vel.angular.z
            cmd_vel.linear.x = linear
            self.cmd_vel_pub.publish(cmd_vel)
            rospy.logfatal("close to the target")
        else:
            self.cmd_vel_pub.publish(cmd_vel)
            rospy.loginfo_throttle(1, f"linear:{cmd_vel.linear.x}, angular:{cmd_vel.angular.z}")

    def position_mode(self):
        goal = MoveBaseGoal()
        assert isinstance(self.goal, PoseStamped)
        goal.target_pose.pose = self.goal.pose
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.linear_velocity = 0.1
        goal.angular_velocity = 0.2
        self.client.send_goal(goal)
        self.client.wait_for_result()
        rospy.logfatal(self.client.get_goal_status_text())


if __name__ == "__main__":
    rospy.init_node("transformer_planner")
    planner = TransformerPlanner006()
    rospy.spin()
