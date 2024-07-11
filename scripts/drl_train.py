import os
from omni.isaac.kit import SimulationApp

linux_user = os.getlogin()
config = {
    "headless": True
}
simulation_app = SimulationApp(config)

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
import re
import time
import angles
import sys
import torch
import tqdm
import argparse

sys.path.append(f"/home/{os.getlogin()}/isaac_sim_ws/devel/lib/python3/dist-packages")

# gym
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

# ROS
import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from gymnasium.utils import seeding
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from isaac_sim.msg import PlanAction, PlanGoal, PlanResult
from tf2_ros import TransformBroadcaster
from actionlib_msgs.msg import GoalStatus
from std_srvs.srv import Empty
from deep_learning_planner.msg import RewardFunction

# Stable-baseline3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

# customer code
from utils.parameters import *
from utils.pose_utils import PoseUtils, get_yaw, get_roll, get_pitch, calculate_geodesic_distance
from transformer_drl_network import TransformerFeatureExtractor, CustomActorCriticPolicy
from utils.sb3_callbacks import CustomCallback
from drl_algorithm import CustomPPO


class TrainingState(enum.Enum):
    REACHED = 0,
    COLLISION = 1,
    TRUNCATED = 2,
    TRAINING = 4


class Environment(gym.Env):
    def __init__(self, scene, total_timestep):
        # Random seeds
        self.seed()

        # const
        self.robot_frame = "base_link"
        self.map_frame = "map"
        self.laser_pool_capacity = interval * laser_length
        self.total_timestep = total_timestep
        self.scene = scene

        # isaac
        self._setup_scene()

        # reinforcement learning global variable
        self.num_envs = 1
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=float)
        self.observation_space = spaces.Dict({
            "laser": spaces.Box(low=0., high=1.0, shape=(laser_length, laser_shape), dtype=float),
            "global_plan": spaces.Box(low=-1., high=1., shape=(look_ahead_poses, 3), dtype=float),
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
        self.pbar = tqdm.tqdm(total=self.total_timestep)
        self.last_pose = None
        self.robot_trapped = 0
        self.last_geodesic_distance = 0
        self.robot_pose = PoseStamped()

        # ros
        self._map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_callback, queue_size=1)
        self._plan_client = SimpleActionClient("/plan", PlanAction)
        self._pose_pub = rospy.Publisher("/robot_pose", PoseStamped, queue_size=1)
        self._scan_pub = rospy.Publisher("/scan", LaserScan, queue_size=1)
        self._clear_costmap_client = rospy.ServiceProxy("/clear", Empty)
        self._goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=1)
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self._reward_pub = rospy.Publisher("/reward", RewardFunction, queue_size=1)
        self._tf_br = TransformBroadcaster()

    # Observation, Reward, terminated, truncated, info
    def step(self, action):
        self.pbar.update()
        if np.nan in action:
            rospy.logfatal(f"neural network calculated an unexpected value(nan)")
        forward, angular = action[0], action[1]
        cmd_vel = Twist()
        cmd_vel.linear.x = forward
        cmd_vel.angular.z = angular
        self._cmd_vel_pub.publish(cmd_vel)
        self.robot.apply_wheel_actions(self.controller.forward(command=np.array([forward, angular])))
        self.world.step()
        self.robot_pose = self._make_robot_pose()
        self._publish_tf()
        state = self._is_done()
        observations = self._get_observation()
        reward, info = self._get_reward(action, observations)
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
        self.robot_pose = self._make_robot_pose()
        self._publish_tf()
        self._clear_costmap_client()

        observations = self._get_observation()

        # reset variables
        self.last_geodesic_distance = calculate_geodesic_distance(self.global_plan)
        self.last_pose = init_pose
        self.training_state = TrainingState.TRAINING
        self.num_iterations = 0
        self.robot_trapped = 0
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
        start_pose = self._make_robot_pose()
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

        # global plan, the plan is in robot frame
        plan = self._make_plan()
        if plan is None:
            rospy.logerr("could not get a global plan")
            self._clear_costmap_client()
        else:
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

    def _get_reward(self, action, observation):
        reward_func = RewardFunction()
        linear, angular = action[0], action[1]

        ##################################################################
        # arrival reward
        digress_distance = np.linalg.norm(self.global_plan[0, :2])
        if self.training_state == TrainingState.REACHED:
            reward_arrival = r_arrival
        elif self.training_state == TrainingState.TRUNCATED:
            reward_arrival = -r_arrival
        else:
            # reward_arrival = r_waypoint * (digress_threshold / 2 - digress_distance)
            geodesic_distance = calculate_geodesic_distance(self.global_plan)
            reward_arrival = r_waypoint * (self.last_geodesic_distance - geodesic_distance)
            self.last_geodesic_distance = geodesic_distance
        ##################################################################
        # collision reward
        reward_collision = 0
        if self.training_state == TrainingState.COLLISION:
            reward_collision = r_collision
        else:
            min_distance = np.min(self.laser_pool[-1])
            if min_distance < 2 * robot_radius:
                reward_collision = r_scan * (2 * robot_radius - min_distance)
        ##################################################################
        # angular reward
        # reward_angular = 0
        # if abs(angular) >= w_thresh:
        #     reward_angular = abs(angular) * r_rotation
        reward_angular = 0
        ##################################################################
        # direction reward
        # desire_angle = np.mean(self.global_plan[:look_ahead, 2])
        # reward_direction = (angle_threshold - abs(desire_angle)) * r_angle
        reward_direction = 0
        ##################################################################
        reward_func.arrival_reward = reward_arrival
        reward_func.collision_reward = reward_collision
        reward_func.angular_reward = reward_angular
        reward_func.direction_reward = reward_direction
        reward_func.total_reward = reward_arrival + reward_collision + reward_angular + reward_direction
        reward_info = dict(arrival=reward_arrival, collision=reward_collision,
                           angular=reward_angular, direction=reward_direction,
                           reward=reward_func.total_reward, is_success=(self.training_state == TrainingState.REACHED))
        self._reward_pub.publish(reward_func)
        return reward_func.total_reward, reward_info

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

        # 2) Robot Trapped or out of bound or get rolled?
        if (abs(get_pitch(curr_pose.pose.orientation)) > math.pi / 3 or
                abs(get_roll(curr_pose.pose.orientation)) > math.pi / 3):
            self.training_state = TrainingState.COLLISION
            rospy.logfatal("Robot get rolled")
            return self.training_state
        if (self._check_out_of_bound(curr_pose.pose.position.x, curr_pose.pose.position.y) or
                self._check_in_unknown_area(curr_pose.pose.position.x, curr_pose.pose.position.y)):
            self.training_state = TrainingState.COLLISION
            rospy.logfatal("Robot out of bound or in an known area")
            return self.training_state
        delta_distance = math.sqrt(math.pow(self.last_pose.pose.position.x - curr_pose.pose.position.x, 2) +
                                   math.pow(self.last_pose.pose.position.y - curr_pose.pose.position.y, 2))
        delta_angle = abs(angles.normalize_angle(get_yaw(curr_pose.pose.orientation) -
                                                 get_yaw(self.last_pose.pose.orientation)))
        if (delta_distance <= 0.01 and delta_angle <= 0.01) or min(self.laser_pool[-1]) < 0.4:
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
        digress_distance = np.min(np.linalg.norm(self.global_plan[:, :2], axis=1))
        if self.num_iterations > max_iteration or digress_distance >= digress_threshold:
            self.training_state = TrainingState.TRUNCATED
            rospy.logfatal("Over the max iteration or digressing the global path a lot before going to the goal")
            return self.training_state
        return self.training_state

    # Callbacks
    def _map_callback(self, map_msg: OccupancyGrid):
        self.map = map_msg

    def _make_robot_pose(self):
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

    def _check_in_unknown_area(self, x, y):
        y_index = int((y - self.map.info.origin.position.y) / self.map.info.resolution)
        x_index = int((x - self.map.info.origin.position.x) / self.map.info.resolution)
        value = self.map.data[x_index + y_index * self.map.info.width]
        return value != 0

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
        # asset_path = assets_root_path + "/Isaac/Robots/Carter/nova_carter.usd"
        wheel_dof_names = ["left_wheel", "right_wheel"]
        # wheel_dof_names = ["joint_wheel_left", "joint_wheel_right"]
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Carters/Carter_0",
                name="Carter_0",
                wheel_dof_names=wheel_dof_names,
                create_robot=True,
                usd_path=asset_path,
                position=np.array([1.5, 1.5, 0.3]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
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
        # self.controller = DifferentialController(name="simple_control", wheel_radius=0.14, wheel_base=0.4132)
        env_usd_path = f"/home/{linux_user}/isaac_sim_ws/src/isaac_sim/isaac/{self.scene}.usd"
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


def load_network_parameters(net_model, model_file, deterministic):
    param = torch.load(model_file)["model_state_dict"]
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
    net_model.features_extractor.load_state_dict(feature_extractor_param)
    net_model.mlp_extractor.mlp_extractor_actor.load_state_dict(mlp_extractor_param)
    net_model.action_net.load_state_dict(action_net_param)
    net_model.features_extractor.requires_grad_(False)
    if deterministic:
        net_model.requires_grad_(False)


if __name__ == "__main__":
    rospy.init_node('drl_training', log_level=rospy.INFO)
    parser = argparse.ArgumentParser(description="this is a script to train the drl based motion planner")
    save_log_dir = f"/home/{os.getlogin()}/isaac_sim_ws/src/deep_learning_planner/rl_logs/runs"
    pretrained_model = "/home/gr-agv-lx91/isaac_sim_ws/src/deep_learning_planner/transformer_logs/model10/best.pth"
    parser.add_argument("--scene", default="small_warehouse", type=str, help="name of training scene")
    parser.add_argument("--step", default=500_000, type=int, help="total time steps for drl training")
    parser.add_argument("--save_logdir", default=save_log_dir, type=str, help="directory to save log")
    parser.add_argument("--model_file", default=pretrained_model, type=str, help="path of model file")
    parser.add_argument("--deterministic", default=False, type=bool, help="training or calculate baseline reward")
    args = parser.parse_args()
    if not os.path.exists(args.save_logdir):
        os.makedirs(args.save_logdir)

    env = Monitor(Environment(args.scene, args.step), args.save_logdir)
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256], qf=[128]),
        # optimizer_kwargs=dict(weight_decay=0.00001),
    )
    CustomActorCriticPolicy.deterministic = args.deterministic
    model = CustomPPO(target_value=200,
                      policy=CustomActorCriticPolicy,
                      env=env,
                      verbose=2,
                      learning_rate=3e-4,
                      batch_size=512,
                      tensorboard_log=args.save_logdir,
                      n_epochs=10,
                      n_steps=1024,
                      gamma=0.99,
                      policy_kwargs=policy_kwargs,
                      device=torch.device("cuda:1"))
    # load_network_parameters(model.policy, args.model_file, args.deterministic)
    model.load_state_dict(args.model_file, args.deterministic)
    save_reward_callback = CustomCallback(check_freq=1024, log_dir=args.save_logdir)
    callback_list = CallbackList([save_reward_callback])
    model.learn(total_timesteps=args.step,
                log_interval=1,
                tb_log_name='drl_policy',
                reset_num_timesteps=True,
                callback=callback_list)
    env.close()
