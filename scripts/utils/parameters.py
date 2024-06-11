import math

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
goal_radius = 0.5
deceleration_tolerance = 1.0

# reinforcement learning train
max_iteration = 512

r_arrival = 50
r_waypoint = 3.2
r_collision = -20
r_scan = -0.2
r_rotation = -0.1
r_angle = 0.6
angle_threshold = math.pi / 9
w_thresh = 0.7
digress_threshold = 1.0

look_ahead = 5

physics_dt = 0.05
rendering_dt = 0.05
