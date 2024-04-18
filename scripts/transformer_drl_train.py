# utils
import math
import os
import re
import enum

import angles
import numpy as np
import tqdm

# gym
import gymnasium as gym
import gymnasium.spaces as spaces

# ROS
import rospy
import torch
from actionlib import SimpleActionClient
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from gymnasium.utils import seeding
from isaac_sim.msg import ResetPosesAction, ResetPosesGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import Empty
from std_srvs.srv import Empty
from deep_learning_planner.msg import RewardFunction

# Stable-baseline3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.ppo import PPO

# package
from transformer_drl_network import CustomActorCriticPolicy, TransformerFeatureExtractor
from pose_utils import PoseUtils, get_yaw
from parameters import *

save_log_dir = f"/home/{os.getlogin()}/isaac_sim_ws/src/deep_learning_planner/rl_logs/runs"
pretrained_model = "/home/gr-agv-lx91/isaac_sim_ws/src/deep_learning_planner/transformer_logs/model7/best.pth"
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        self._pose_utils = PoseUtils(robot_radius)

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

        # ros
        self._plan_sub = rospy.Subscriber("/move_base/GlobalPlanner/robot_frame_plan",
                                          Path, self._path_callback, queue_size=1)
        self._map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_callback, queue_size=1)
        self._laser_sub = rospy.Subscriber("/scan", LaserScan, self._laser_callback, queue_size=1)
        self._init_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self._goal_pub = SimpleActionClient("/move_base", MoveBaseAction)
        self._reward_pub = rospy.Publisher("/reward", RewardFunction, queue_size=1)
        self._imu_sub = rospy.Subscriber("/imu", Imu, self._imu_callback, queue_size=1)

        # isaac sim
        self._pause_client = rospy.ServiceProxy("/pause", Empty)
        self._unpause_client = rospy.ServiceProxy("/unpause", Empty)
        self._close_client = rospy.ServiceProxy("/close", Empty)
        self._reset_client = SimpleActionClient("/reset", ResetPosesAction)

        self._wait_for_map()

    # Observation, Reward, terminated, truncated, info
    def step(self, action):
        self._unpause()
        self._take_action(action)
        observations = self._get_observation()
        self._pause()
        state = self._is_done()
        reward = self._get_reward(action)
        return (observations, reward, state == TrainingState.REACHED,
                state == TrainingState.COLLISION or state == TrainingState.TRUNCATED, {})

    def reset(self, **kwargs):
        rospy.loginfo("resetting!")
        isaac_init_poses = ResetPosesGoal()
        self._cmd_vel_pub.publish(Twist())

        init_pose = self._pose_utils.get_random_pose(self.map, self.map_frame, self.map_frame)
        init_pose_world = self._pose_utils.transform_pose(init_pose, self.map_frame)
        isaac_init_poses.poses.append(init_pose_world)
        isaac_init_poses.prefix.append(0)

        # reset robot pose in isaac sim
        self._reset_client.wait_for_server()
        self._cmd_vel_pub.publish(Twist())
        self._reset_client.send_goal_and_wait(isaac_init_poses)
        rospy.sleep(3.0)

        # reset robot pose in ros amcl
        amcl_init_pose = PoseWithCovarianceStamped()
        amcl_init_pose.header = init_pose.header
        amcl_init_pose.header.stamp = rospy.Time.now()
        amcl_init_pose.pose.pose = init_pose.pose
        self._init_pub.publish(amcl_init_pose)
        rospy.sleep(2.0)

        # publish new goal
        self.target = self._pose_utils.get_random_pose(self.map, self.map_frame, self.map_frame)
        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose = self.target
        self._goal_pub.send_goal(move_base_goal)
        rospy.sleep(3.0)

        # reset variables
        self.last_pose = init_pose_world
        self.training_state = TrainingState.TRAINING
        self.num_iterations = 0
        self.collision_times = 0
        self.robot_trapped = 0
        self.pbar.reset()

        return self._get_observation(), {}

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
                rospy.sleep(2.0)
            else:
                break

    def _get_observation(self):
        pose = self._pose_utils.transform_pose(self.target, self.robot_frame)
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
        return {
            "laser": np.random.rand(6, 1080),
            "global_plan": global_plan,
            "goal": pos
        }

    def _get_reward(self, action):
        reward_msg = RewardFunction()
        reward = 0
        linear = action[0] * (max_vel_x - min_vel_x) + min_vel_x
        angular = action[1] * (max_vel_z - min_vel_z) + min_vel_z

        if self.training_state == TrainingState.REACHED:
            reward += goal_reached_reward
            reward_msg.goal_reach_reward = goal_reached_reward

        if self.training_state == TrainingState.COLLISION:
            reward += collision_punish
            reward_msg.collision_punish = -collision_punish
        else:
            min_obstacle_distance = min(self.laser_pool[-1].ranges)
            if min_obstacle_distance < 2 * robot_radius:
                reward -= (2 * robot_radius - min_obstacle_distance) * obstacle_punish_weight
                reward_msg.collision_punish = -(2 * robot_radius - min_obstacle_distance) * obstacle_punish_weight

        reward += linear * abs(velocity_reward_weight)
        reward -= abs(angular) * angular_punish_weight
        reward -= abs(self.imu.linear_acceleration.x) * imu_punish_weight
        reward_msg.linear_vel_reward = linear * abs(velocity_reward_weight)
        reward_msg.angular_vel_punish = -abs(angular) * angular_punish_weight
        reward_msg.imu_punish = -abs(self.imu.linear_acceleration.x) * imu_punish_weight
        reward_msg.total_reward = reward
        self._reward_pub.publish(reward_msg)
        return reward

    def _take_action(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0] * (max_vel_x - min_vel_x) + min_vel_x
        cmd_vel.angular.z = action[1] * (max_vel_z - min_vel_z) + min_vel_z
        self._cmd_vel_pub.publish(cmd_vel)
        rospy.sleep(0.05)

    def _is_done(self):
        self.pbar.update()
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
            self._cmd_vel_pub.publish(Twist())
            rospy.logfatal("Carter reached the goal!")
            return self.training_state

        # 2) Robot Trapped?
        delta_distance = math.sqrt(math.pow(self.last_pose.pose.position.x - curr_pose.pose.position.x, 2) +
                                   math.pow(self.last_pose.pose.position.y - curr_pose.pose.position.y, 2))
        delta_angle = abs(angles.normalize_angle(get_yaw(curr_pose.pose.orientation) -
                                                 get_yaw(self.last_pose.pose.orientation)))
        if delta_distance <= 0.05 and delta_angle <= 0.05:
            self.robot_trapped += 1
        else:
            self.robot_trapped = 0
            self.last_pose = curr_pose
        if self.robot_trapped >= 5:
            self.training_state = TrainingState.COLLISION
            self._cmd_vel_pub.publish(Twist())
            rospy.logfatal("Collision")
            return self.training_state

        # 3) maximum number of iterations?
        self.num_iterations += 1
        if self.num_iterations > max_iteration:
            self.training_state = TrainingState.TRUNCATED
            self._cmd_vel_pub.publish(Twist())
            rospy.logfatal("Over the max iteration before going to the goal")
            return self.training_state
        return self.training_state

    # Callbacks
    def _map_callback(self, map_msg: OccupancyGrid):
        self.map = map_msg

    def _laser_callback(self, laser_msg: LaserScan):
        if len(self.laser_pool) > self.laser_pool_capacity:
            self.laser_pool.pop(0)
        self.laser_pool.append(laser_msg)

    def _path_callback(self, global_path_msg):
        self.global_plan = global_path_msg

    def _imu_callback(self, msg: Imu):
        self.imu = msg

    # Service Clients
    def _pause(self):
        self._pause_client.call()
        rospy.logdebug("pause isaac sim")

    def _unpause(self):
        self._unpause_client.call()
        rospy.logdebug("unpause isaac sim")


def load_network_parameters(net_model):
    param = torch.load(pretrained_model)["model_state_dict"]
    feature_extractor_param = {}
    mlp_extractor_param = {}
    value_net_param = {}
    action_net_param = {}
    for k, v in param.items():
        if re.search("^module.action_net.", k):
            new_k = re.sub("^module.action_net.[0-9].", "", k)
            action_net_param[new_k] = v
        elif re.search("^module.value_net.", k):
            new_k = re.sub("^module.value_net.[0-9].", "", k)
            value_net_param[new_k] = v
        elif re.search("^module.mlp_extractor_(actor|critic).", k):
            new_k = k.replace("module.", "")
            mlp_extractor_param[new_k] = v
        elif re.search("^module.", k):
            new_k = k.replace("module.", "")
            feature_extractor_param[new_k] = v
    # do not update parameter in feature extractor
    # net_model.features_extractor.requires_grad_(True)
    net_model.action_net.load_state_dict(action_net_param)
    net_model.value_net.load_state_dict(value_net_param)
    net_model.features_extractor.load_state_dict(feature_extractor_param)
    net_model.mlp_extractor.load_state_dict(mlp_extractor_param)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True


if __name__ == "__main__":
    rospy.init_node('simple_rl_training', log_level=rospy.INFO)
    env = Monitor(SimpleEnv(), save_log_dir)
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256], vf=[128]),
        log_std_init=-1.5
    )
    model = PPO(CustomActorCriticPolicy,  # ?
                env=env,
                verbose=2,  # similar to log level
                learning_rate=5e-6,
                batch_size=256,
                tensorboard_log=save_log_dir,
                n_epochs=10,  # run n_epochs after collecting a roll-out buffer to optimize parameters
                n_steps=256,  # the size of roll-out buffer
                gamma=0.99,  # discount factors
                policy_kwargs=policy_kwargs,
                device="cuda")
    load_network_parameters(model.policy)
    save_model_callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=save_log_dir, verbose=2)
    callback_list = CallbackList([save_model_callback])
    model.learn(total_timesteps=200000,
                log_interval=5,
                tb_log_name='drl_policy',
                reset_num_timesteps=True,
                callback=callback_list)
    env.close()
