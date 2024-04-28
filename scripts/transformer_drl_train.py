# utils
import enum
import math
import os
import re
import time
import typing
import angles
import sys

sys.path.append(f"/home/{os.getlogin()}/isaac_sim_ws/devel/lib/python3/dist-packages")
# gym
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
# ROS
import rospy
import torch
import tqdm
from actionlib import SimpleActionClient
from deep_learning_planner.msg import RewardFunction
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from gymnasium.utils import seeding
from isaac_sim.msg import ResetPosesAction, ResetPosesGoal, ResetPosesResult, StepAction, StepGoal, StepResult
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan, Imu
# Stable-baseline3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import SACPolicy
from std_msgs.msg import Empty
from std_srvs.srv import Empty

from parameters import *
from pose_utils import PoseUtils, get_yaw, get_distance_from_path
from transformer_drl_network import TransformerFeatureExtractor
from sb3_callbacks import SaveOnBestTrainingRewardCallback, RewardCallback

save_log_dir = f"/home/{os.getlogin()}/isaac_sim_ws/src/deep_learning_planner/rl_logs/runs"
pretrained_model = "/home/gr-agv-lx91/isaac_sim_ws/src/deep_learning_planner/transformer_logs/model9/best.pth"
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
scene = "hospital"


class TrainingState(enum.Enum):
    REACHED = 0,
    COLLISION = 1,
    TRUNCATED = 2,
    TRAINING = 4


class SimpleEnv(gym.Env):
    def __init__(self):
        # Random seeds
        self.seed()

        # const
        self.robot_frame = "base_link"
        self.map_frame = "map"
        self.laser_pool_capacity = interval * laser_length

        # global variable
        self.num_envs = 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=float)
        self.observation_space = spaces.Dict({
            "laser": spaces.Box(low=0., high=1.0, shape=(6, 1080), dtype=float),
            "global_plan": spaces.Box(low=-1., high=1., shape=(20, 3), dtype=float),
            "goal": spaces.Box(low=-math.inf, high=math.inf, shape=(3,), dtype=float)
        })

        # utils
        self._pose_utils = PoseUtils(robot_radius, scene=scene)

        # private variable
        self.global_plan = Path()
        self.map = OccupancyGrid()
        self.imu = Imu()
        self.laser_pool = []
        self.training_state = TrainingState.TRAINING
        self.num_iterations = 0
        self.target = PoseStamped()
        self.collision_times = 0
        self.pbar = tqdm.tqdm(total=max_iteration)
        self.last_pose = None
        self.robot_trapped = 0
        self.last_global_plan_length = None
        self.explore_punish_pool = []
        self.entire_distance = 0
        self.last_geodesic_distance = 0

        # ros
        global_plan_topic = "/move_base/GlobalPlanner/robot_frame_plan"
        self._plan_sub = rospy.Subscriber(global_plan_topic, Path, self._path_callback, queue_size=1)
        self._map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_callback, queue_size=1)
        self._laser_sub = rospy.Subscriber("/scan", LaserScan, self._laser_callback, queue_size=10)
        self._init_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        self._goal_pub = SimpleActionClient("/move_base", MoveBaseAction)
        self._reward_pub = rospy.Publisher("/reward", RewardFunction, queue_size=1)
        self._imu_sub = rospy.Subscriber("/imu", Imu, self._imu_callback, queue_size=10)

        # isaac sim
        self._close_client = rospy.ServiceProxy("/close", Empty)
        self._reset_client = SimpleActionClient("/reset", ResetPosesAction)
        self._step_client = SimpleActionClient("/step", StepAction)
        self._wait_for_map()

    # Observation, Reward, terminated, truncated, info
    def step(self, action):
        self.pbar.update()
        start_time = rospy.Time.now()
        result = self._call_step(action[0], action[1])
        observations = self._get_observation(result)
        state = self._is_done()
        reward, info = self._get_drl_vo_reward(action, observations)
        end_time = rospy.Time.now()
        rospy.logfatal(f"step time: {(end_time - start_time).to_sec()}(s)")
        return (observations, reward, state == TrainingState.REACHED,
                state == TrainingState.COLLISION or state == TrainingState.TRUNCATED, info)

    def reset(self, **kwargs):
        rospy.loginfo("resetting!")
        isaac_init_poses = ResetPosesGoal()
        init_pose, self.target = self._pose_utils.get_preset_pose(self.map_frame)

        init_pose_world = self._pose_utils.transform_pose(init_pose, self.map_frame)
        isaac_init_poses.poses.append(init_pose_world)
        isaac_init_poses.prefix.append(0)

        while True:
            # reset robot pose in isaac sim
            self._reset_client.wait_for_server()
            self._reset_client.send_goal_and_wait(isaac_init_poses)
            result = self._reset_client.get_result()
            time.sleep(3.0)

            # reset robot pose in ros amcl
            amcl_init_pose = PoseWithCovarianceStamped()
            amcl_init_pose.header = init_pose.header
            amcl_init_pose.header.stamp = rospy.Time.now()
            amcl_init_pose.pose.pose = init_pose.pose
            self._init_pub.publish(amcl_init_pose)
            time.sleep(3.0)

            # publish new goal
            move_base_goal = MoveBaseGoal()
            move_base_goal.target_pose = self.target
            self._goal_pub.send_goal(move_base_goal)

            self.global_plan = None
            total_time = 0
            flag = True

            while self.global_plan is None:
                time.sleep(0.5)
                total_time += 0.5
                if total_time > 10.0:
                    rospy.logerr("get global plan over 10s, resetting again")
                    flag = False
                    break
            if flag:
                break

        # reset variables
        # noinspection PyTypeChecker
        self.entire_distance = get_distance_from_path(self.global_plan)
        self.last_geodesic_distance = self.entire_distance
        self.last_pose = init_pose_world
        self.training_state = TrainingState.TRAINING
        self.num_iterations = 0
        self.collision_times = 0
        self.robot_trapped = 0
        self.last_global_plan_length = None
        self.explore_punish_pool.clear()
        self.pbar.reset()

        return self._get_observation(result), {}

    def close(self):
        rospy.logfatal("Closing IsaacSim Simulator")
        self._close_client.call()
        self.pbar.close()
        rospy.signal_shutdown("Closing ROS Signal")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _wait_for_map(self):
        while True:
            if self.map.info.resolution == 0:
                rospy.logfatal("robot do not receive map yet")
                time.sleep(2.0)
            else:
                break

    def _get_observation(self, result: typing.Union[StepResult, ResetPosesResult]):
        # goal
        pose = self._pose_utils.transform_pose(self.target, self.robot_frame)
        assert isinstance(pose, PoseStamped)
        goal = np.array((pose.pose.position.x, pose.pose.position.y, get_yaw(pose.pose.orientation)))

        # global plan
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
                padding = np.stack([goal for _ in range(look_ahead_poses - len(global_plan))], axis=0)
                global_plan = np.concatenate([global_plan, padding], axis=0)
        else:
            global_plan = np.stack([goal for _ in range(look_ahead_poses)], axis=0)

        # lasers
        lasers = np.array(result.ranges)

        # normalization
        global_plan = global_plan / np.array([look_ahead_distance, look_ahead_distance, np.pi])
        pos = goal / np.array([look_ahead_distance, look_ahead_distance, np.pi])
        lasers = np.reshape(lasers, (result.frames, laser_shape)) / laser_range

        return {
            "laser": lasers,
            "global_plan": global_plan,
            "goal": goal
        }

    def _get_reward(self, action, observation):
        reward_msg = RewardFunction()
        global_plan = observation["global_plan"]
        global_plan_length = len(self.global_plan.poses)
        reward = 0
        linear, angular = action

        # goal reach reward
        if self.training_state == TrainingState.REACHED:
            reward += goal_reached_reward
            reward_msg.goal_reach_reward = goal_reached_reward
        elif self.training_state == TrainingState.TRUNCATED:
            not_reach_punish = -get_distance_from_path(self.global_plan) / self.entire_distance * goal_reached_reward
            reward += not_reach_punish
            reward_msg.goal_reach_reward = not_reach_punish

        # collision or close to obstacle punish
        if self.training_state == TrainingState.COLLISION:
            reward -= collision_punish
            reward_msg.collision_punish = -collision_punish
        else:
            min_obstacle_distance = min(self.laser_pool[-1])
            if min_obstacle_distance < 2 * robot_radius:
                reward -= (2 * robot_radius - min_obstacle_distance) * obstacle_punish_weight
                reward_msg.collision_punish = -(2 * robot_radius - min_obstacle_distance) * obstacle_punish_weight

        # move forward reward
        if self.last_global_plan_length is None:
            reward_msg.move_forward_reward = 0.0
        else:
            delta_length = self.last_global_plan_length - global_plan_length
            reward_msg.move_forward_reward = -pave_punish if delta_length <= 0 else move_forward_reward
            reward += reward_msg.move_forward_reward

        # linear velocity reward
        reward += abs(linear) * velocity_reward_weight
        reward_msg.linear_velocity_reward = abs(linear) * velocity_reward_weight

        # angular velocity reward
        (x, y, yaw) = global_plan[look_ahead]
        angle = math.atan2(y, x)
        if math.fabs(angle) <= math.pi / 6:
            reward -= angular * angular_punish_weight
            reward_msg.angular_vel_punish = angular * angular_punish_weight

        # follow global path reward
        distance = np.linalg.norm(global_plan[:, :2], axis=1)
        min_distance = np.min(distance) * look_ahead_distance
        if min_distance < 0.1:
            reward += follow_reward
            reward_msg.follow_reward = follow_reward
        else:
            reward -= min_distance * unfollow_punish_weight
            reward_msg.follow_reward = -min_distance * unfollow_punish_weight

        self.last_global_plan_length = global_plan_length
        reward_msg.total_reward = reward
        self._reward_pub.publish(reward_msg)
        return reward

    def _get_drl_vo_reward(self, action, observation):
        info = {}
        r_arrival = 20
        r_waypoint = 3.2
        r_collision = -20
        r_scan = -0.2
        r_rotation = -0.1

        w_thresh = 0.7

        reward = 0
        # linear = action[0] * (max_vel_x - min_vel_x) + min_vel_x
        # angular = action[1] * (max_vel_z - min_vel_z) + min_vel_z
        linear, angular = action
        laser = observation["laser"]

        # reach the goal reward
        if self.training_state == TrainingState.REACHED:
            reward_arrival = r_arrival
        elif self.training_state == TrainingState.TRUNCATED:
            reward_arrival = -r_arrival
        else:
            geodesic_distance = get_distance_from_path(self.global_plan)
            reward_arrival = r_waypoint * (self.last_geodesic_distance - geodesic_distance)
            self.last_geodesic_distance = geodesic_distance
        reward += reward_arrival
        info["arrival"] = reward_arrival

        # obstacle reward
        reward_collision = 0
        if self.training_state == TrainingState.COLLISION:
            reward_collision = r_collision
        else:
            min_distance = np.min(laser[-1])
            if min_distance < 3 * robot_radius:
                reward_collision = r_scan * (3 * robot_radius - min_distance)
        info["collision"] = reward_collision
        reward += reward_collision

        # angular velocity punish
        info["angular"] = 0
        if abs(angular) >= w_thresh:
            reward = abs(angular) * r_rotation
            info["angular"] = abs(angular) * r_rotation

        # theta reward
        return reward, info

    def _is_done(self):
        # 1) Goal reached?
        curr_pose = self._pose_utils.get_robot_pose(self.robot_frame, self.map_frame)
        dist_to_goal = np.linalg.norm(
            np.array([
                curr_pose.pose.position.x - self.target.pose.position.x,
                curr_pose.pose.position.y - self.target.pose.position.y,
                curr_pose.pose.position.z - self.target.pose.position.z
            ])
        )
        if dist_to_goal <= goal_radius:
            self.training_state = TrainingState.REACHED
            rospy.logfatal("Carter reached the goal!")
            return self.training_state

        # 2) Robot Trapped?
        delta_distance = math.sqrt(math.pow(self.last_pose.pose.position.x - curr_pose.pose.position.x, 2) +
                                   math.pow(self.last_pose.pose.position.y - curr_pose.pose.position.y, 2))
        delta_angle = abs(angles.normalize_angle(get_yaw(curr_pose.pose.orientation) -
                                                 get_yaw(self.last_pose.pose.orientation)))
        if delta_distance <= 0.05 and delta_angle <= 0.05 and min(self.laser_pool[-1]) <= robot_radius:
            self.robot_trapped += 1
        else:
            self.robot_trapped = 0
            self.last_pose = curr_pose
        if self.robot_trapped >= 5:
            self.training_state = TrainingState.COLLISION
            rospy.logfatal("Collision")
            return self.training_state

        # 3) maximum number of iterations?
        self.num_iterations += 1
        if self.num_iterations > max_iteration:
            self.training_state = TrainingState.TRUNCATED
            rospy.logfatal("Over the max iteration before going to the goal")
            return self.training_state
        return self.training_state

    # Callbacks
    def _map_callback(self, map_msg: OccupancyGrid):
        self.map = map_msg

    def _laser_callback(self, laser_msg: LaserScan):
        if len(self.laser_pool) > self.laser_pool_capacity:
            self.laser_pool.pop(0)
        self.laser_pool.append(laser_msg.ranges)

    def _path_callback(self, global_path_msg):
        self.global_plan = global_path_msg

    def _imu_callback(self, msg: Imu):
        self.imu = msg

    def _call_step(self, forward, angular):
        goal = StepGoal()
        goal.forward = forward
        goal.angular = angular
        self._step_client.send_goal_and_wait(goal)
        return self._step_client.get_result()


def load_network_parameters(net_model):
    param = torch.load(pretrained_model)["model_state_dict"]
    feature_extractor_param = {}
    mlp_extractor_param = {}
    action_net_param = {}
    for k, v in param.items():
        if re.search("^module.policy_net.", k):
            new_k = re.sub("^module.policy_net.[0-9].", "", k)
            action_net_param[new_k] = v
        elif re.search("^module.mlp_extractor_policy.", k):
            new_k = k.replace("module.mlp_extractor_policy.", "")
            mlp_extractor_param[new_k] = v
        else:
            new_k = k.replace("module.", "")
            feature_extractor_param[new_k] = v
    net_model.actor.features_extractor.load_state_dict(feature_extractor_param)
    net_model.actor.latent_pi.load_state_dict(mlp_extractor_param)
    net_model.actor.mu.load_state_dict(action_net_param)
    net_model.critic.features_extractor.load_state_dict(feature_extractor_param)


if __name__ == "__main__":
    rospy.init_node('simple_rl_training', log_level=rospy.INFO)
    env = Monitor(SimpleEnv(), save_log_dir)
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256], qf=[128]),
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(
            weight_decay=0.00001
        )
    )
    model = SAC(SACPolicy,
                env=env,
                buffer_size=100_000,
                learning_rate=1e-5,
                batch_size=256,
                learning_starts=0,
                tensorboard_log=save_log_dir,
                policy_kwargs=policy_kwargs,
                device="cuda")
    load_network_parameters(model.policy)
    save_model_callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=save_log_dir, verbose=2)
    reward_callback = RewardCallback(verbose=2)
    callback_list = CallbackList([save_model_callback, reward_callback])
    model.learn(total_timesteps=200000,
                log_interval=5,
                tb_log_name='drl_policy',
                reset_num_timesteps=True,
                callback=callback_list)
    env.close()
