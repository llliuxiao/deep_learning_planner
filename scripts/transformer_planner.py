# utils
import argparse
import os
import numpy as np

# ros
import rospy
import tf2_geometry_msgs
import tf2_ros

# torch
import torch
from einops import repeat
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from stable_baselines3.ppo import PPO
from tf.transformations import euler_from_quaternion
from tf2_ros import Buffer
from tf2_ros.transform_listener import TransformListener

# customer module
from utils.parameters import *
from utils.pose_utils import distance_points2d
from transformer_network import RobotTransformer


class TransformerPlanner:
    def __init__(self, flag_, model_file_, robot_frame, scan_topic_name,
                 global_topic_name, goal_topic_name, cmd_topic_name, velocity_factor=1.0):
        self.flag = flag_
        self.model_file = model_file_

        self.robot_frame = robot_frame
        self.velocity_factor = velocity_factor
        self.device = torch.device("cuda:0")
        if self.flag == "imitation":
            self.model = RobotTransformer()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            param = torch.load(model_file_)["model_state_dict"]
            self.model.load_state_dict(param)
            self.model.train(False)
        elif self.flag == "reinforcement":
            self.model = PPO.load(model_file_, device=self.device)
        else:
            rospy.logerr("The flag is neither imitation nor reinforcement")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.goal = None
        self.global_plan = None
        self.goal_reached = True

        # make laser pool as a queue
        self.laser_pool = []
        self.laser_pool_capacity = interval * laser_length
        self.last_cmd_vel = (0.0, 0.0)

        self.scan_sub = rospy.Subscriber(scan_topic_name, LaserScan, self.laser_callback, queue_size=1)
        self.global_plan_sub = rospy.Subscriber(global_topic_name, Path, self.global_plan_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber(goal_topic_name, PoseStamped, self.goal_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher(cmd_topic_name, Twist, queue_size=1)

        self.turning_distance = 0
        self.looking_ahead_distance = 1.0

        # 50 HZ
        rospy.Timer(rospy.Duration(secs=0, nsecs=20000000), self.cmd_inference)

    def laser_callback(self, msg: LaserScan):
        if len(self.laser_pool) > self.laser_pool_capacity:
            self.laser_pool.pop(0)
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        self.laser_pool.append(ranges)

    def goal_callback(self, msg: PoseStamped):
        if self.goal is None:
            self.goal_reached = False
            rospy.logfatal("receive new goal")
        else:
            distance = math.sqrt((msg.pose.position.x - self.goal.pose.position.x) ** 2 +
                                 (msg.pose.position.y - self.goal.pose.position.y) ** 2)
            if distance > 0.05:
                self.goal_reached = False
                rospy.logfatal("receive new goal")
        self.goal = msg

    def global_plan_callback(self, msg: Path):
        self.global_plan = msg

    @staticmethod
    def linear_deceleration_stopping(v0, x, d) -> float:
        return math.sqrt((d - x) / d) * v0

    @staticmethod
    def linear_deceleration_turning(v0, x, d, target) -> float:
        # d mean total distance, x mean how much distance left
        return math.sqrt(x / d * (v0 ** 2) + (d - x) / d * (target ** 2))

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

    def make_tensor(self):
        # laser
        lasers = [self.laser_pool[-1] for _ in range(laser_length)]
        for i in range(laser_length - 1, 0, -1):
            prefix = len(self.laser_pool) - i * interval
            if prefix < 0:
                lasers[laser_length - i - 1] = self.laser_pool[0]
            else:
                lasers[laser_length - i - 1] = self.laser_pool[prefix]
        laser_tensor = torch.tensor(np.array(lasers), dtype=torch.float)
        laser_tensor = torch.div(laser_tensor, torch.tensor(laser_range, dtype=torch.float))

        # goal
        scale = torch.tensor([look_ahead_distance, look_ahead_distance, torch.pi], dtype=torch.float)
        goal = tf2_geometry_msgs.PoseStamped()
        goal.pose = self.goal.pose
        goal.header.stamp = rospy.Time(0)
        goal.header.frame_id = "map"
        try:
            target_pose = self.tf_buffer.transform(goal, self.robot_frame)
            assert isinstance(target_pose, PoseStamped)
        except tf2_ros.TransformException as ex:
            rospy.logfatal(ex)
            rospy.logfatal("could not transform goal to the robot frame")
            return
        goal = (target_pose.pose.position.x, target_pose.pose.position.y, self._get_yaw(target_pose.pose.orientation))
        goal_tensor = torch.tensor(goal, dtype=torch.float)
        goal_tensor = torch.div(goal_tensor, scale)

        # global plan
        assert isinstance(self.global_plan, Path)
        global_plan = []
        for pose in self.global_plan.poses:
            assert isinstance(pose, PoseStamped)
            x = pose.pose.position.x
            y = pose.pose.position.y
            yaw = self._get_yaw(pose.pose.orientation)
            global_plan.append((x, y, yaw))
        global_plan = torch.tensor(global_plan, dtype=torch.float)
        if len(global_plan) > 0:
            global_plan = global_plan[:min(len(global_plan), look_ahead_poses * down_sample):down_sample, :]
            if len(global_plan) < look_ahead_poses:
                padding = repeat(goal_tensor, "d -> b d", b=look_ahead_poses - len(global_plan))
                global_plan = torch.concat([global_plan, padding])
        else:
            global_plan = repeat(goal_tensor, "d -> b d", b=look_ahead_poses)
        global_plan_tensor = torch.div(global_plan, scale)

        # laser mask
        laser_mask = torch.ones((1, laser_length, laser_length), dtype=torch.bool).triu(1).to(self.device)

        if self.flag == "imitation":
            laser_tensor = torch.unsqueeze(laser_tensor, 0).to(self.device)
            global_plan_tensor = torch.unsqueeze(global_plan_tensor, 0).to(self.device)
            goal_tensor = torch.unsqueeze(goal_tensor, 0).to(self.device)
            return laser_tensor, global_plan_tensor, goal_tensor, laser_mask
        else:
            # unsqueeze
            return laser_tensor, global_plan_tensor, goal_tensor, laser_mask

    def wait_for_raw_data(self):
        while True:
            if self.global_plan is not None and len(self.laser_pool) > 0:
                return
            else:
                rospy.sleep(0.5)

    def inference(self, laser: torch.Tensor, global_plan: torch.Tensor, goal: torch.Tensor, mask=None):
        if self.flag == "imitation":
            with torch.no_grad():
                action = self.model(laser, global_plan, goal, mask)
            action = torch.squeeze(action)
            return self.velocity_factor * action[0].item(), self.velocity_factor * action[1].item()
        elif self.flag == "reinforcement":
            action, _states = self.model.predict({
                "laser": laser.cpu().numpy(),
                "global_plan": global_plan.cpu().numpy(),
                "goal": goal.cpu().numpy()
            })
            action = np.squeeze(action)
            return action[0], action[1]
        else:
            rospy.logerr("The flag is neither imitation nor reinforcement")

    def smooth(self, forward, angular):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.5 * forward + 0.5 * self.last_cmd_vel[0]
        cmd_vel.angular.z = 0.5 * angular + 0.5 * self.last_cmd_vel[1]
        rospy.loginfo_throttle(1, f"linear:{cmd_vel.linear.x}, angular:{cmd_vel.angular.z}")
        return cmd_vel

    def is_turing(self):
        for pose in self.global_plan.poses:
            delta_angle = self._get_yaw(pose.pose.orientation)
            distance_turning = distance_points2d(self.global_plan.poses[0].pose.position, pose.pose.position)
            if distance_turning > self.looking_ahead_distance:
                return False
            elif math.fabs(delta_angle) >= math.pi / 4:
                self.turning_distance = distance_turning
                return True

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
        action = self.inference(laser_tensor, global_plan_tensor, goal_tensor, laser_mask)
        cmd_vel.linear.x, cmd_vel.angular.z = action[0], action[1]
        distance = self.get_distance_to_goal()
        assert isinstance(self.global_plan, Path)
        if distance <= goal_radius and len(self.global_plan.poses) < 50:
            self.goal_reached = True
            self.goal = None
            self.cmd_vel_pub.publish(Twist())
            rospy.logfatal("reach the goal")
        elif distance <= deceleration_tolerance and len(self.global_plan.poses) < 50:
            linear = self.linear_deceleration_stopping(1.0 * self.velocity_factor,
                                                       deceleration_tolerance - distance,
                                                       deceleration_tolerance - goal_radius)
            cmd_vel.angular.z = linear / cmd_vel.linear.x * cmd_vel.angular.z
            cmd_vel.linear.x = linear
            self.cmd_vel_pub.publish(cmd_vel)
        elif self.is_turing() and self.flag == "reinforcement":
            linear = self.linear_deceleration_turning(1.0, self.turning_distance, self.looking_ahead_distance, 0.6)
            cmd_vel.linear.x = linear
            self.cmd_vel_pub.publish(cmd_vel)
        else:
            cmd_vel = self.smooth(cmd_vel.linear.x, cmd_vel.angular.z)
            self.cmd_vel_pub.publish(cmd_vel)
        self.last_cmd_vel = (cmd_vel.linear.x, cmd_vel.angular.z)
        rospy.loginfo(f"x: {cmd_vel.linear.x}, z: {cmd_vel.angular.z}")

    @staticmethod
    def _get_yaw(quaternion: Quaternion):
        _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return yaw


if __name__ == "__main__":
    rospy.init_node("transformer_planner")
    parse = argparse.ArgumentParser(description="A ROS motion planner to deploy dnn model")
    root_path = f"/home/{os.getlogin()}/isaac_sim_ws/src/deep_learning_planner"
    imitation_file = os.path.join(root_path, "transformer_logs/model9/best.pth")
    reinforcement_file = os.path.join(root_path, "rl_logs/runs/drl_policy_31/best_model.zip")
    parse.add_argument("-m", "--mode", type=str, default="imitation", help="imitation or reinforcement")
    parse.add_argument("-r", "--robot", type=str, default="sim", help="sim or gr-agv234")
    args = parse.parse_args()
    flag = args.mode
    model_file = imitation_file if flag == "imitation" else reinforcement_file
    if args.robot == "sim":
        planner_kwargs = dict(
            flag_=flag,
            model_file_=model_file,
            robot_frame="base_link",
            scan_topic_name="/scan",
            global_topic_name="/move_base/GlobalPlanner/robot_frame_plan",
            goal_topic_name="/move_base/current_goal",
            cmd_topic_name="/cmd_vel",
        )
    else:
        planner_kwargs = dict(
            flag_=flag,
            model_file_=model_file,
            robot_frame="base_footprint",
            scan_topic_name="/map_scan",
            global_topic_name="/robot4/move_base/GlobalPlanner/robot_frame_plan",
            goal_topic_name="/robot4/move_base/current_goal",
            cmd_topic_name="/vm_gsd601/gr_canopen_vm_motor/mobile_base_controller/cmd_vel"
        )
    planner = TransformerPlanner(**planner_kwargs)
    rospy.spin()
