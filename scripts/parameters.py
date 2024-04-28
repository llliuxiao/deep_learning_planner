# robot parameters
max_vel_x = 1.0
min_vel_x = 0.0
max_vel_z = 1.0
min_vel_z = -1.0
robot_radius = 0.5

# laser
laser_length = 6
interval = 10
laser_range = 10.0
laser_shape = 1080

# global_path
down_sample = 4
look_ahead_poses = 20
look_ahead_distance = look_ahead_poses * down_sample * 0.05 + 1.0

# goal
goal_radius = 0.3
deceleration_tolerance = 1.0

# reinforcement learning train
max_iteration = 1024

goal_reached_reward = 1000
velocity_reward_weight = 1.0
pave_punish = 1.0
move_forward_reward = 1.0
collision_punish = 1000
angular_punish_weight = 2.0
obstacle_punish_weight = 5.0
imu_punish_weight = 0.5
follow_reward = 1.0
unfollow_punish_weight = 5.0

look_ahead = 5

physics_dt = 0.05
rendering_dt = 0.05
