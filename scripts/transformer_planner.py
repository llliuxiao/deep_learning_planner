# torch
import torch
from einops import repeat

# utils
import math

# ros
import rospy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from tf2_ros import Buffer
from tf2_ros.transform_listener import TransformListener

from transformer_network import RobotTransformer

model_file = "/home/gr-agv-lx91/isaac_sim_ws/src/supervised_learning_planner/transformer_logs/model3/best.pth"
laser_length = 6
interval = 10
down_sample = 4
look_ahead_poses = 20

goal_tolerance = 0.3
deceleration_tolerance = 1.0
robot_frame = "base_footprint"


class TransformerPlanner:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = RobotTransformer()
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        param = torch.load(model_file)["model_state_dict"]
        self.model.load_state_dict(param)
        self.model.train(False)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.goal = None
        self.global_plan = None
        self.goal_reached = True

        # make laser pool as a queue
        self.laser_pool = []
        self.laser_pool_capacity = interval * laser_length

        self.scan_sub = rospy.Subscriber("/map_scan", LaserScan, self.laser_callback, queue_size=1)
        self.global_plan_sub = rospy.Subscriber("/move_base/GlobalPlanner/robot_frame_plan", Path,
                                                self.global_plan_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber("/move_base/current_goal", PoseStamped, self.goal_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/vm_gsd601/gr_canopen_vm_motor/mobile_base_controller/cmd_vel", Twist,
                                           queue_size=1)
        # 25 HZ
        rospy.Timer(rospy.Duration(secs=0, nsecs=40000000), self.cmd_inference)

    def laser_callback(self, msg: LaserScan):
        if len(self.laser_pool) > self.laser_pool_capacity:
            self.laser_pool.pop(0)
        self.laser_pool.append(msg.ranges)

    def goal_callback(self, msg: PoseStamped):
        if self.goal is None:
            self.goal_reached = False
        else:
            distance = math.sqrt((msg.pose.position.x - self.goal.pose.position.x) ** 2 +
                                 (msg.pose.position.y - self.goal.pose.position.y) ** 2)
            if distance > 0.05:
                self.goal_reached = False
        self.goal = msg

    def global_plan_callback(self, msg: Path):
        self.global_plan = msg

    def is_done(self):
        try:
            robot_pose = self.tf_buffer.lookup_transform("map", robot_frame, rospy.Time(0))
            assert isinstance(robot_pose, TransformStamped)
        except tf2_ros.TransformException:
            rospy.logfatal("could not get robot pose")
            return False
        assert isinstance(self.goal, PoseStamped)
        delta_x = self.goal.pose.position.x - robot_pose.transform.translation.x
        delta_y = self.goal.pose.position.y - robot_pose.transform.translation.y
        # delta_angle = self._get_yaw(self.goal.pose.orientation) - self._get_yaw(robot_pose.transform.rotation)
        # delta_angle = math.fabs(angles.normalize_angle(delta_angle))
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        if distance <= goal_tolerance:
            self.goal_reached = True
            self.cmd_vel_pub.publish(Twist())
            rospy.logfatal("reach the goal")
            return True
        elif distance <= deceleration_tolerance:
            self.linear_deceleration(1.0,
                                     deceleration_tolerance - distance,
                                     deceleration_tolerance - goal_tolerance)
            rospy.logfatal("close to the target")
            return True
        else:
            return False

    def linear_deceleration(self, v0, x, d):
        cmd = Twist()
        cmd.linear.x = math.sqrt((d - x) / d) * v0
        self.cmd_vel_pub.publish(cmd)

    def make_tensor(self):
        # laser
        lasers = [self.laser_pool[-1] for _ in range(laser_length)]
        for i in range(laser_length - 1, 0, -1):
            prefix = len(self.laser_pool) - i * interval
            if prefix < 0:
                continue
            else:
                lasers[laser_length - i - 1] = self.laser_pool[prefix]
        laser_tensor = torch.tensor(lasers, dtype=torch.float)

        # goal
        goal = tf2_geometry_msgs.PoseStamped()
        goal.pose = self.goal.pose
        goal.header.stamp = rospy.Time(0)
        goal.header.frame_id = "map"
        try:
            target_pose = self.tf_buffer.transform(goal, robot_frame)
            assert isinstance(target_pose, PoseStamped)
        except tf2_ros.TransformException as ex:
            rospy.logfatal(ex)
            rospy.logfatal("could not transform goal to the robot frame")
            return
        goal = (target_pose.pose.position.x, target_pose.pose.position.y, self._get_yaw(target_pose.pose.orientation))
        goal_tensor = torch.tensor(goal, dtype=torch.float)

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
        global_plan_tensor = global_plan

        # laser mask
        laser_mask = torch.ones((laser_length, laser_length), dtype=torch.bool).triu(1)
        laser_mask = torch.stack([laser_mask for _ in range(torch.cuda.device_count())]).to(self.device)

        # unsqueeze
        laser_tensor = torch.unsqueeze(laser_tensor, 0).to(self.device)
        global_plan_tensor = torch.unsqueeze(global_plan_tensor, 0).to(self.device)
        goal_tensor = torch.unsqueeze(goal_tensor, 0).to(self.device)
        return laser_tensor, global_plan_tensor, goal_tensor, laser_mask

    def wait_for_raw_data(self):
        while True:
            if self.global_plan is not None and len(self.laser_pool) > 0:
                return
            else:
                rospy.sleep(0.5)

    def cmd_inference(self, event):
        if self.goal_reached or self.is_done():
            return
        cmd_vel = Twist()
        self.wait_for_raw_data()
        tensor = self.make_tensor()
        if tensor is None:
            return
        else:
            laser_tensor, global_plan_tensor, goal_tensor, laser_mask = tensor
        self.model.train(False)
        with torch.no_grad():
            predict = self.model(laser_tensor, global_plan_tensor, goal_tensor, laser_mask)
        predict = torch.squeeze(predict)
        cmd_vel.linear.x = predict[0].item()
        cmd_vel.angular.z = predict[1].item()
        rospy.loginfo_throttle(1, f"linear:{cmd_vel.linear.x}, angular:{cmd_vel.angular.z}")
        self.cmd_vel_pub.publish(cmd_vel)

    @staticmethod
    def _get_yaw(quaternion: Quaternion):
        _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return yaw


if __name__ == "__main__":
    rospy.init_node("transformer_planner")
    planner = TransformerPlanner()
    rospy.spin()
