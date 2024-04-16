# robot parameters
max_vel_x = 1.0
min_vel_x = 0.0
max_vel_z = 1.0
min_vel_z = -1.0
robot_radius = 0.45

# laser
laser_length = 6
interval = 10
laser_range = 10.0

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
collision_punish = -1000
angular_punish_weight = 2.0
obstacle_punish_weight = 10.0
