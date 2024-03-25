"""
A python script to record trajectory information tuple for pretraining
Goal pose, cmd_vel and global path are recorded in a json file
while laser scan are recorded in a binary file by numpy(.npy)
"""
# utils
import math
import os
import random
import sys
import threading
import copy
import enum
import json
import numpy as np

# ROS
import rospy
import tf2_geometry_msgs
from actionlib import SimpleActionClient
from geometry_msgs.msg import Pose, Twist, TwistStamped
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from isaac_sim.msg import ResetPosesGoal, ResetPosesAction
from move_base_msgs.msg import MoveBaseGoal, MoveBaseResult, MoveBaseAction
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException
from tqdm import tqdm
import message_filters

# robot parameters:
robot_radius = 0.3
goal_radius = 0.3
angle_threshold = 0.5
max_trapped_time = 10.0

# global const
max_step = 2000
linux_user = os.getlogin()
dataset_root_path = f"/home/{linux_user}/Downloads/pretraining_dataset"
global_path_subsample = 4


class RobotState(enum.Enum):
    REACHED = 1
    RUNNING = 2
    TRAPPED = 4


class TrajectoryRecorder:
    def __init__(self):
        # To reset Simulations
        rospy.logdebug("START init trajectory recorder")

        # visual
        self.pbar = tqdm(total=max_step)

        # pool
        self.trajectory_num = 0
        self.step_num = 0
        self.dataset_pool = []
        self.dataset_info = {}

        # ros tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # write threading
        self.dataset_thread = threading.Thread(target=self._writing_thread)
        self.dataset_condition_lock = threading.Condition()
        self.close_signal = False

        # ros communication
        self.scan_sub = message_filters.Subscriber("/scan", LaserScan)
        self.cmd_vel_sub = message_filters.Subscriber("/cmd_vel_stamped", TwistStamped)
        self.msg_filter = message_filters.TimeSynchronizer([self.scan_sub, self.cmd_vel_sub], 10)
        self.msg_filter.registerCallback(self._sensor_callback)

        self.close_client = rospy.ServiceProxy("/close", Empty)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_callback)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)
        self.reset_client = SimpleActionClient("/reset", ResetPosesAction)
        self.nav_client = SimpleActionClient("move_base", MoveBaseAction)
        self.path_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self._path_callback, queue_size=1)

        # global path
        self.global_path = []

        # Poses
        self.start_pose = Pose()
        self.target_pose = Pose()

        # flags
        self._goal_reached = False
        self.state = RobotState.RUNNING

        self.dataset_thread.start()
        self.reset(0)
        rospy.logfatal("Finished Init trajectory recorder")

    def _writing_thread(self):
        while not self.close_signal:
            self.dataset_condition_lock.acquire()
            if len(self.dataset_pool) == 0:
                self.dataset_condition_lock.wait()
            dataset = copy.deepcopy(self.dataset_pool)
            self.dataset_pool.clear()
            self.dataset_condition_lock.release()
            for (laser, path) in dataset:
                np.save(path, np.array(laser))
            dataset.clear()
            del dataset

    def _sensor_callback(self, scan_msg: LaserScan, cmd_vel_msg: TwistStamped):
        try:  # try to transform global target pose to the robot base_link
            point = tf2_geometry_msgs.PoseStamped()
            point.pose = self.target_pose
            point.header.frame_id = "map"
            target_pose = self.tf_buffer.transform(point, "base_link")
            assert isinstance(target_pose, PoseStamped)
        except TransformException:
            rospy.logfatal("could not transform target to robot base_link")
            return
        except AssertionError:
            rospy.logfatal("the type of return of tf buffer transform is not PoseStamped, system exits modify it now!")
            sys.exit(-1)

        laser_path = os.path.join(f"{dataset_root_path}/trajectory{self.trajectory_num}", f"step{self.step_num}.npy")
        data = {
            "time": rospy.Time.now().to_sec(),
            "target_x": target_pose.pose.position.x,
            "target_y": target_pose.pose.position.y,
            "target_yaw": self._get_yaw(target_pose.pose.orientation),
            "cmd_vel_linear": cmd_vel_msg.twist.linear.x,
            "cmd_vel_angular": cmd_vel_msg.twist.angular.z,
            "laser_path": laser_path
        }
        self.dataset_info[f"trajectory{self.trajectory_num}"]["data"].append(data)
        self.dataset_condition_lock.acquire()
        self.dataset_pool.append((scan_msg.ranges, laser_path))
        self.dataset_condition_lock.notify()
        self.dataset_condition_lock.release()
        self.step_num += 1
        self.pbar.update()

    def reset(self, trajectory_num):
        rospy.logdebug("Start initializing robot...")
        # reset pbar description
        self.pbar.set_description(f"trajectory{trajectory_num}")
        # set step num
        self.trajectory_num = trajectory_num
        self.step_num = 0

        if os.path.exists(f"{dataset_root_path}/trajectory{trajectory_num}"):
            os.system(f"rm -rf {dataset_root_path}/trajectory{trajectory_num}")
        os.mkdir(f"{dataset_root_path}/trajectory{trajectory_num}")

        # reset robot pose in isaac sim
        self.start_pose = self._get_random_pos_on_map(self.map)
        reset_goal = ResetPosesGoal()
        pose = PoseStamped()
        pose.pose = self.start_pose
        reset_goal.poses.append(pose)
        reset_goal.prefix.append(0)
        self.reset_client.wait_for_server()
        self.reset_client.send_goal(reset_goal)
        self.reset_client.wait_for_result()

        # wait for isaac sim reset robot position
        rospy.sleep(2.0)

        # reset robot pose for amcl
        self._pub_initial_position(self.start_pose)

        # wait for amcl updating localization
        rospy.sleep(2.0)
        self.state = RobotState.RUNNING

        # get a random goal where the target distance is between 5-10m
        while True:
            self.target_pose = self._get_random_pos_on_map(self.map)
            dx = self.target_pose.position.x - self.start_pose.position.x
            dy = self.target_pose.position.y - self.start_pose.position.y
            if 2 <= math.sqrt(dx ** 2 + dy ** 2) <= 100:
                break
        self.dataset_info[f"trajectory{trajectory_num}"] = {
            "start_x": self.start_pose.position.x,
            "start_y": self.start_pose.position.y,
            "start_yaw": self._get_yaw(self.start_pose.orientation),
            "target_x": self.target_pose.position.x,
            "target_y": self.target_pose.position.x,
            "target_yaw": self._get_yaw(self.target_pose.orientation),
            "data": []
        }
        self._publish_goal_position(self.target_pose)

        # wait for global planner calculating a global path
        rospy.sleep(4.0)
        self.dataset_info[f"trajectory{trajectory_num}"]["global_path"] = self.global_path
        rospy.logdebug("Finish initializing robot...")

    def close(self):
        rospy.logfatal("Closing IsaacSim")
        with open(os.path.join(dataset_root_path, "dataset_info.json"), "w") as file:
            json.dump(self.dataset_info, fp=file)
        self.pbar.close()
        self.close_client.call()

        # close writing thread
        self.dataset_condition_lock.acquire()
        self.close_signal = True
        self.dataset_condition_lock.notify()
        self.dataset_condition_lock.release()
        self.dataset_thread.join()

    def _map_callback(self, map_msg):
        self.map = map_msg

    def _pub_initial_position(self, pose: Pose):
        inital_pose = PoseWithCovarianceStamped()
        inital_pose.header.frame_id = "map"
        inital_pose.header.stamp = rospy.Time.now()
        inital_pose.pose.pose = pose
        self.initial_pose_pub.publish(inital_pose)

    def _publish_goal_position(self, pose: Pose):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose = pose
        self.nav_client.send_goal(goal, done_cb=self._nav_callback_done)

    def _get_random_pos_on_map(self, grid_map) -> Pose:
        pose = Pose()
        map_width = grid_map.info.width * grid_map.info.resolution + grid_map.info.origin.position.x
        map_height = grid_map.info.height * grid_map.info.resolution + grid_map.info.origin.position.y
        x = random.uniform(0.0, map_width)
        y = random.uniform(0.0, map_height)
        while not self._is_pos_valid(x, y, robot_radius, grid_map):
            x = random.uniform(0.0, map_width)
            y = random.uniform(0.0, map_height)
        theta = random.uniform(-math.pi, math.pi)
        pose.position.x = x
        pose.position.y = y
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = np.sin(theta / 2)
        pose.orientation.w = np.cos(theta / 2)
        return pose

    @staticmethod
    def _is_pos_valid(x, y, radius, grid_map):
        cell_radius = int(radius / grid_map.info.resolution)
        y_index = int((y - grid_map.info.origin.position.y) / grid_map.info.resolution)
        x_index = int((x - grid_map.info.origin.position.x) / grid_map.info.resolution)

        for i in range(x_index - cell_radius, x_index + cell_radius):
            for j in range(y_index - cell_radius, y_index + cell_radius):
                index = j * grid_map.info.width + i
                if index >= len(grid_map.data):
                    return False
                try:
                    val = grid_map.data[index]
                except IndexError:
                    print(f"IndexError: index: {index}, map_length: {len(grid_map.data)}")
                    return False
                if val != 0:
                    return False
        return True

    @staticmethod
    def _get_yaw(quaternion: Quaternion):
        _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return yaw

    def clear_collision_trajectory(self, trajectory_num):
        del self.dataset_info[f"trajectory{trajectory_num}"]

    def reset_pbar(self, last_step):
        self.pbar.reset(total=max_step)
        self.pbar.update(last_step)

    # Goal State Code
    # uint8 PENDING         = 0
    # uint8 ACTIVE          = 1
    # uint8 PREEMPTED       = 2
    # uint8 SUCCEEDED       = 3
    # uint8 ABORTED         = 4
    # uint8 REJECTED        = 5
    # uint8 PREEMPTING      = 6
    # uint8 RECALLING       = 7
    # uint8 RECALLED        = 8
    # uint8 LOST            = 9
    def _nav_callback_done(self, state: int, result: MoveBaseResult):
        if state == 3:
            self.state = RobotState.REACHED
        elif state == 4 or self == 5:
            self.state = RobotState.TRAPPED
        else:
            self.state = RobotState.RUNNING

    def _path_callback(self, msg: Path):
        self.global_path.clear()
        containers = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            yaw = self._get_yaw(pose.pose.orientation)
            containers.append((x, y, yaw))
        self.global_path = containers[::global_path_subsample]
        if len(containers) % global_path_subsample != 0:
            self.global_path.append(containers[-1])


if __name__ == "__main__":
    rospy.init_node("trajectory_recorder")
    recorder = TrajectoryRecorder()
    num = 0
    step = 0
    while step < max_step:
        while True:
            if recorder.state == RobotState.REACHED:
                rospy.loginfo("\nReach the goal")
                rospy.sleep(2.0)
                num += 1
                step += recorder.step_num
                break
            elif recorder.state == RobotState.TRAPPED:
                rospy.logfatal("\nTrapped")
                rospy.sleep(2.0)
                recorder.clear_collision_trajectory(num)
                recorder.reset_pbar(step)
                break
            rospy.sleep(0.05)
        if step < max_step:
            recorder.reset(num)
    recorder.close()
    print("shut down")
