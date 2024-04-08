# utils
import math
import os
import re

# gym
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
# ROS
import rospy
import torch
from actionlib import SimpleActionClient
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from gymnasium.utils import seeding
from isaac_sim.msg import ResetPosesAction, ResetPosesGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.ppo import PPO
from std_msgs.msg import Empty
from std_srvs.srv import Empty

from transformer_network import RobotTransformer
from pose_utils import PoseUtils, get_yaw
import tqdm

# robot parameters:
max_vel = 1.0
min_vel = 0.0
max_ang = 1.0
min_ang = -1.0
robot_radius = 0.45
goal_radius = 0.5
max_trapped_time = 5.0

max_iteration = 1024

save_log_dir = f"/home/{os.getlogin()}/isaac_sim_ws/src/reinforcement_learning_planner/logs/runs"
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)


class SimpleEnv(gym.Env):
    def __init__(self):
        # Random seeds
        self.seed()

        # const
        self.robot_frame = "base_link"
        self.map_frame = "map"

        # global variable
        self.num_envs = 1
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=float)
        self.observation_space = spaces.Dict({
            "laser": spaces.Box(low=0.0, high=1.0, shape=(900,), dtype=float),
            "global_plan": spaces.Box(low=0.0, high=100, shape=(20,3), dtype=float),
            "goal": spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        })

        # utils
        self._pose_utils = PoseUtils(robot_radius)

        # private variable
        self.global_plan = Path()
        self.map = OccupancyGrid()
        self.laser = LaserScan()
        self.terminated = False
        self.truncated = False
        self.num_iterations = 0
        self.target = PoseStamped()
        self.collision_times = 0
        self.pbar = tqdm.tqdm(total=max_iteration)

        # ros
        self._plan_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self._path_callback, queue_size=1)
        self._map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_callback, queue_size=1)
        self._laser_sub = rospy.Subscriber("/scan", LaserScan, self._laser_callback, queue_size=1)
        self._init_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self._goal_pub = SimpleActionClient("/move_base", MoveBaseAction)

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
        done = self._is_done()
        reward = self._get_reward(action)
        terminated, truncated = self._get_done()
        return observations, reward, terminated, truncated, {}

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
        self.terminated = False
        self.truncated = False
        self.num_iterations = 0
        self.collision_times = 0
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

    # to reinforce works by ETH in 2017, only laser and target pose are required
    def _get_observation(self):
        pose = self._pose_utils.transform_pose(self.target, self.robot_frame)
        assert isinstance(pose, PoseStamped)
        pos = (pose.pose.position.x, pose.pose.position.y, get_yaw(pose.pose.orientation))
        obs = {"laser": np.array(self.laser.ranges) / 10.0, "goal": np.array(pos) / np.array((10.0, 10.0, math.pi))}
        return obs

    def _get_reward(self, action):
        reward = 0
        r_arrival = 50

        # reach or collision
        if self.terminated:
            reward += r_arrival
        elif self.truncated and self.num_iterations >= max_iteration:
            reward += 10

        # close to target

        # angular velocity
        if abs(action[1]) > 0.7:
            reward -= abs(action[1]) * 0.2

        # collision
        if self.truncated and self.num_iterations < max_iteration:
            reward -= r_arrival
        else:
            min_dist = min(self.laser.ranges)
            if min_dist <= 3 * robot_radius:
                reward -= (3 * robot_radius - min_dist) * 0.2

        return reward

    def _get_done(self):
        return self.terminated, self.truncated

    def _take_action(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
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
            self.terminated = True
            self._cmd_vel_pub.publish(Twist())
            rospy.logfatal("Carter reached the goal!")
            return True

        # 2) Robot Trapped?
        laser_ranges = np.array(self.laser.ranges)
        collision_points = np.where(laser_ranges <= robot_radius)[0]
        if len(collision_points) > 50:
            self.collision_times += 1
            rospy.logwarn_throttle(1, f"collision ++ = {self.collision_times}")
        else:
            self.collision_times = 0
        if self.collision_times > 10:
            self.truncated = True
            self._cmd_vel_pub.publish(Twist())
            rospy.logfatal("Collision")
            return True

        # 3) maximum number of iterations?
        self.num_iterations += 1
        if self.num_iterations > max_iteration:
            self.truncated = True
            self._cmd_vel_pub.publish(Twist())
            rospy.logfatal("Over the max iteration before going to the goal")
            return True
        return False

    # Callbacks
    def _map_callback(self, map_msg: OccupancyGrid):
        self.map = map_msg

    def _laser_callback(self, laser_msg: LaserScan):
        self.laser = laser_msg

    def _path_callback(self, global_path_msg):
        self.global_plan = global_path_msg

    # Service Clients
    def _pause(self):
        self._pause_client.call()
        rospy.logdebug("pause isaac sim")

    def _unpause(self):
        self._unpause_client.call()
        rospy.logdebug("unpause isaac sim")


def load_network_parameters(net_model):
    file = "/home/gr-agv-lx91/isaac_sim_ws/src/supervised_learning_planner/logs/model3/best.pth"
    param = torch.load(file)["model_state_dict"]
    feature_extractor_param = {}
    mlp_extractor_param = {}
    value_net_param = {}
    action_net_param = {}
    for k, v in param.items():
        if re.search("^module.action_net.", k):
            new_k = k.replace("module.action_net.", "")
            action_net_param[new_k] = v
        elif re.search("^module.value_net.", k):
            new_k = k.replace("module.value_net.", "")
            value_net_param[new_k] = v
        elif re.search("^module.actor.", k):
            new_k = k.replace("module.actor.", "policy_net.")
            mlp_extractor_param[new_k] = v
        elif re.search("^module.critic.", k):
            new_k = k.replace("module.critic.", "value_net.")
            mlp_extractor_param[new_k] = v
        elif re.search("^module.", k):
            new_k = k.replace("module.", "")
            feature_extractor_param[new_k] = v
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
        features_extractor_class=RobotTransformer,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(pi=[512], vf=[256])
    )
    model = PPO("MultiInputPolicy",  # ?
                env=env,
                verbose=2,  # similar to log level
                learning_rate=5e-5,
                batch_size=256,
                tensorboard_log=save_log_dir,
                n_epochs=10,  # run n_epochs after collecting a roll-out buffer to optimize parameters
                n_steps=max_iteration,  # the size of roll-out buffer
                gamma=0.99,  # discount factors
                policy_kwargs=policy_kwargs,
                device=torch.device("cuda"))
    load_network_parameters(model.policy)
    save_model_callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=save_log_dir, verbose=2)
    callback_list = CallbackList([])
    model.learn(total_timesteps=1000000,
                log_interval=5,
                tb_log_name='drl_policy',
                reset_num_timesteps=True,
                callback=callback_list)
    env.close()
