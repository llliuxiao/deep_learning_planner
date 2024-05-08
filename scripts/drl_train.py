import os

from omni.isaac.kit import SimulationApp

linux_user = os.getlogin()
CARTER_USD_PATH = f"/home/{linux_user}/isaac_sim_ws/src/isaac_sim/isaac/carter.usd"
config = {
    "headless": True
}
simulation_app = SimulationApp(config)

# utils

# isaac
from omni.isaac.core import World
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import RotatingLidarPhysX
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.core.utils.nucleus import get_assets_root_path

# utils
import enum
import math
import os
import re
import time
import angles
import sys
import torch
import tqdm

sys.path.append(f"/home/{os.getlogin()}/isaac_sim_ws/devel/lib/python3/dist-packages")

# gym
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

# ROS
import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, TransformStamped
from gymnasium.utils import seeding
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from isaac_sim.msg import PlanAction, PlanGoal, PlanResult
from tf2_ros import TransformBroadcaster
from actionlib_msgs.msg import GoalStatus
from std_srvs.srv import Empty

# Stable-baseline3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import SACPolicy

# customer code
from parameters import *
from pose_utils import PoseUtils, get_yaw, calculate_geodesic_distance
from transformer_drl_network import TransformerFeatureExtractor
from sb3_callbacks import SaveOnBestTrainingRewardCallback, RewardCallback

save_log_dir = f"/home/{os.getlogin()}/isaac_sim_ws/src/deep_learning_planner/rl_logs/runs"
pretrained_model = "/home/gr-agv-lx91/isaac_sim_ws/src/deep_learning_planner/transformer_logs/model9/best.pth"
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)

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

        # isaac
        self._setup_scene()

        # const
        self.robot_frame = "base_link"
        self.map_frame = "map"
        self.laser_pool_capacity = interval * laser_length

        # reinforcement learning global variable
        self.num_envs = 1
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=float)
        self.observation_space = spaces.Dict({
            "laser": spaces.Box(low=0., high=1.0, shape=(6, 1080), dtype=float),
            "global_plan": spaces.Box(low=-1., high=1., shape=(20, 3), dtype=float),
            "goal": spaces.Box(low=-math.inf, high=math.inf, shape=(3,), dtype=float)
        })

        # utils
        self._pose_utils = PoseUtils(robot_radius, scene=scene)

        # private variable
        self.map = OccupancyGrid()
        self.laser_pool = []
        self.training_state = TrainingState.TRAINING
        self.num_iterations = 0
        self.target = PoseStamped()
        self.pbar = tqdm.tqdm(total=max_iteration)
        self.last_pose = None
        self.robot_trapped = 0
        self.last_geodesic_distance = 0
        self.robot_pose = PoseStamped()

        # ros
        self._map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_callback, queue_size=1)
        self._plan_client = SimpleActionClient("/plan", PlanAction)
        self._pose_pub = rospy.Publisher("/robot_pose", PoseStamped, queue_size=1)
        self._scan_pub = rospy.Publisher("/scan", LaserScan, queue_size=1)
        self._clear_costmap_client = rospy.ServiceProxy("/clear_costmap", Empty)
        self._goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=1)
        self._tf_br = TransformBroadcaster()

    # Observation, Reward, terminated, truncated, info
    def step(self, action):
        self.pbar.update()
        if np.nan in action:
            rospy.logfatal(f"neural network calculated an unexpected value(nan)")
        forward = np.clip(action[0], min_vel_x, max_vel_x)
        angular = np.clip(action[1], min_vel_z, max_vel_z)
        self.robot.apply_wheel_actions(self.controller.forward(command=np.array([forward, angular])))
        self.world.step()
        self.robot_pose = self._make_pose()
        self._publish_tf()
        state = self._is_done()
        observations = self._get_observation()
        reward, info = self._get_drl_vo_reward(action, observations)
        return (observations, reward, state == TrainingState.REACHED,
                state == TrainingState.COLLISION or state == TrainingState.TRUNCATED, info)

    def reset(self, **kwargs):
        rospy.loginfo("resetting!")
        self._wait_for_map()

        # clear the buffer before resetting
        self.robot.apply_wheel_actions(self.controller.forward(command=np.array([0.0, 0.0])))
        self.world.step()
        self.world.reset()
        self.laser_pool.clear()

        init_pose, self.target = self._pose_utils.get_preset_pose(self.map_frame)

        # reset robot pose in isaac sim
        x, y, yaw = (init_pose.pose.position.x, init_pose.pose.position.y, get_yaw(init_pose.pose.orientation))
        position = np.array([x, y, 0.3])
        orientation = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.robot.set_world_pose(position=position, orientation=orientation)
        self.world.step()
        time.sleep(2.0)

        # ros
        self.target.header.stamp = rospy.Time.now()
        self.target.header.frame_id = self.map_frame
        self._goal_pub.publish(self.target)
        self.robot_pose = self._make_pose()
        self._publish_tf()
        self._clear_costmap_client()

        observations = self._get_observation()

        # reset variables
        self.last_geodesic_distance = calculate_geodesic_distance(self.global_plan)
        self.last_pose = init_pose
        self.training_state = TrainingState.TRAINING
        self.num_iterations = 0
        self.robot_trapped = 0
        self.pbar.reset()
        return observations, {}

    def close(self):
        rospy.logfatal("Closing IsaacSim Simulator")
        simulation_app.close()
        self.pbar.close()
        rospy.signal_shutdown("Closing ROS Signal")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _make_plan(self):
        goal = PlanGoal()
        goal.target = self.target
        start_pose = self._make_pose()
        start_pose.header.frame_id = self.map_frame
        start_pose.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = self.map_frame
        goal.target.header.stamp = rospy.Time.now()
        goal.start = start_pose
        self._pose_pub.publish(start_pose)
        self._plan_client.send_goal_and_wait(goal)
        if self._plan_client.get_state() == GoalStatus.SUCCEEDED:
            result = self._plan_client.get_result()
            return result

    def _wait_for_map(self):
        while True:
            if self.map.info.resolution == 0:
                rospy.logfatal("robot do not receive map yet")
                time.sleep(2.0)
            else:
                break

    def _get_observation(self):
        # goal
        robot_pose = self.robot_pose
        dx = self.target.pose.position.x - robot_pose.pose.position.x
        dy = self.target.pose.position.y - robot_pose.pose.position.y
        cos_yaw = math.cos(get_yaw(robot_pose.pose.orientation))
        sin_yaw = -math.sin(get_yaw(robot_pose.pose.orientation))
        goal = np.array([
            dx * cos_yaw - dy * sin_yaw,
            dx * sin_yaw + dy * cos_yaw,
            angles.normalize_angle(get_yaw(self.target.pose.orientation) - get_yaw(robot_pose.pose.orientation))
        ])

        # lasers
        self._store_laser()
        lasers = [self.laser_pool[-1] for _ in range(laser_length)]
        for i in range(laser_length - 1, 0, -1):
            prefix = len(self.laser_pool) - i * 10
            if prefix < 0:
                lasers[laser_length - i - 1] = self.laser_pool[0]
            else:
                lasers[laser_length - i - 1] = self.laser_pool[prefix]

        # global plan
        count = 0
        while True:
            plan = self._make_plan()
            if plan is not None:
                break
            else:
                count += 1
                if count >= 10:
                    return {
                        "laser": np.array(lasers) / laser_range,
                        "global_plan": np.zeros(shape=(20, 3)),
                        "goal": goal / np.array([look_ahead_distance, look_ahead_distance, np.pi]),
                    }
                rospy.logerr("could not get a global plan")
                self._clear_costmap_client()
        self.global_plan = np.array([plan.x, plan.y, plan.yaw]).T
        if len(self.global_plan) > 0:
            global_plan = self.global_plan[:min(len(self.global_plan), look_ahead_poses * down_sample):down_sample, :]
            if len(global_plan) < look_ahead_poses:
                padding = np.stack([goal for _ in range(look_ahead_poses - len(global_plan))], axis=0)
                global_plan = np.concatenate([global_plan, padding], axis=0)
        else:
            global_plan = np.stack([goal for _ in range(look_ahead_poses)], axis=0)

        # normalization
        global_plan = global_plan / np.array([look_ahead_distance, look_ahead_distance, np.pi])
        goal = goal / np.array([look_ahead_distance, look_ahead_distance, np.pi])
        lasers = np.array(lasers) / laser_range

        return {
            "laser": lasers,
            "global_plan": global_plan,
            "goal": goal
        }

    def _get_drl_vo_reward(self, action, observation):
        info = {}
        r_arrival = 20
        r_waypoint = 3.2
        r_collision = -20
        r_scan = -0.2
        r_rotation = -0.1

        w_thresh = 0.7

        reward = 0

        linear, angular = action
        laser = observation["laser"]

        # reach the goal reward
        if self.training_state == TrainingState.REACHED:
            reward_arrival = r_arrival
        elif self.training_state == TrainingState.TRUNCATED:
            reward_arrival = -r_arrival
        else:
            geodesic_distance = calculate_geodesic_distance(self.global_plan)
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
        curr_pose = self.robot_pose
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

        # 2) Robot Trapped or out of bound?
        if self._check_out_of_bound(curr_pose.pose.position.x, curr_pose.pose.position.y):
            self.training_state = TrainingState.COLLISION
            rospy.logfatal("Robot out of bound")
            return self.training_state
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

    def _make_pose(self):
        pose = PoseStamped()
        robot_pose = self.robot.get_world_pose()
        pose.pose.position.x = robot_pose[0][0]
        pose.pose.position.y = robot_pose[0][1]
        pose.pose.orientation.w = robot_pose[1][0]
        pose.pose.orientation.x = robot_pose[1][1]
        pose.pose.orientation.y = robot_pose[1][2]
        pose.pose.orientation.z = robot_pose[1][3]
        return pose

    def _store_laser(self):
        cur_laser = self.lidarInterface.get_linear_depth_data(self.lidar_path)[:, 0]
        self._publish_scan(cur_laser)
        if len(self.laser_pool) >= self.laser_pool_capacity:
            self.laser_pool.pop(0)
        self.laser_pool.append(cur_laser)

    # if point is out of bound, return true
    def _check_out_of_bound(self, x, y):
        bound_x = self.map.info.width * self.map.info.resolution + self.map.info.origin.position.x
        bound_y = self.map.info.height * self.map.info.resolution + self.map.info.origin.position.y
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        return not (origin_x <= x <= bound_x and origin_y <= y <= bound_y)

    def _publish_tf(self):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.robot_frame
        transform.transform.translation.x = self.robot_pose.pose.position.x
        transform.transform.translation.y = self.robot_pose.pose.position.y
        transform.transform.translation.z = self.robot_pose.pose.position.z
        transform.transform.rotation = self.robot_pose.pose.orientation
        self._tf_br.sendTransform(transform)

    def _publish_scan(self, ranges):
        scan = LaserScan()
        scan.header.frame_id = self.robot_frame
        scan.header.stamp = rospy.Time.now()
        scan.range_max = 10.0
        scan.range_min = 0.4
        scan.ranges = ranges
        scan.angle_min = - 3 * math.pi / 4
        scan.angle_max = 3 * math.pi / 4
        scan.angle_increment = 0.25 * math.pi / 180
        self._scan_pub.publish(scan)

    def _setup_scene(self):
        self.world = World(stage_units_in_meters=1.0, physics_dt=0.05, rendering_dt=0.05)
        assets_root_path = get_assets_root_path()
        asset_path = assets_root_path + "/Isaac/Robots/Carter/carter_v1.usd"
        wheel_dof_names = ["left_wheel", "right_wheel"]
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Carters/Carter_0",
                name="Carter_0",
                wheel_dof_names=wheel_dof_names,
                create_robot=True,
                usd_path=asset_path,
                position=np.array([1.5, 1.5, 0.3]),
                orientation=np.array([np.cos(1.0), 0.0, 0.0, 0.0])
            )
        )
        self.lidar_path = "/World/Carters/Carter_0/chassis_link/lidar"
        self.carter_lidar = self.world.scene.add(
            RotatingLidarPhysX(
                prim_path=self.lidar_path,
                name="lidar",
                rotation_frequency=0,
                translation=np.array([-0.06, 0, 0.38]),
                fov=(270, 0.0), resolution=(0.25, 0.0),
                valid_range=(0.4, 10.0)
            )
        )
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        self.controller = DifferentialController(name="simple_control", wheel_radius=0.24, wheel_base=0.56)
        env_usd_path = f"/home/{linux_user}/isaac_sim_ws/src/isaac_sim/isaac/hospital.usd"
        add_reference_to_stage(usd_path=env_usd_path, prim_path="/World/Envs/Env_0")
        self.world.scene.add(
            GeometryPrim(
                prim_path="/World/Envs/Env_0",
                name="Env",
                collision=True,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
        )
        self.world.reset()


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
    # net_model.actor.features_extractor.requires_grad_(False)
    # net_model.actor.features_extractor.dense.requires_grad_(True)
    # net_model.critic.features_extractor.requires_grad_(False)
    # net_model.critic.features_extractor.dense.requires_grad_(True)


if __name__ == "__main__":
    rospy.init_node('simple_rl_training', log_level=rospy.INFO)
    env = Monitor(SimpleEnv(), save_log_dir)
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        share_features_extractor=True,
        net_arch=dict(pi=[256], qf=[128]),
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(weight_decay=0.00001),
    )
    model = SAC(SACPolicy,
                env=env,
                buffer_size=200_000,
                learning_rate=5e-6,
                batch_size=256,
                learning_starts=0,
                tensorboard_log=save_log_dir,
                policy_kwargs=policy_kwargs,
                device="cuda:1")
    load_network_parameters(model.policy)
    save_model_callback = SaveOnBestTrainingRewardCallback(check_freq=1024, log_dir=save_log_dir, verbose=2)
    reward_callback = RewardCallback(verbose=2)
    callback_list = CallbackList([save_model_callback, reward_callback])
    model.learn(total_timesteps=500_000,
                log_interval=2,  # episodes interval of log
                tb_log_name='drl_policy',
                reset_num_timesteps=True,
                callback=callback_list)
    env.close()
