# torch
import torch
from network import Model

# utils
import angles
import math

# ros
import rospy
import tf2_ros
from tf2_ros import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, TransformStamped
from sensor_msgs.msg import LaserScan
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

model_file = "/home/gr-agv-lx91/isaac_sim_ws/src/supervised_learning_planner/logs/model/best.pth"


class DeepMotionPlanner:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = Model()
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        param = torch.load(model_file)["model_state_dict"]
        self.model.load_state_dict(param)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber("/move_base/current_goal", PoseStamped, self.goal_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Timer(rospy.Duration(secs=0, nsecs=50), self.cmd_inference)

        self.laser = None
        self.goal = None

    def laser_callback(self, msg: LaserScan):
        self.laser = msg.ranges

    def goal_callback(self, msg: PoseStamped):
        self.goal = msg

    def is_done(self):
        try:
            robot_pose = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0))
            assert isinstance(robot_pose, TransformStamped)
        except tf2_ros.TransformException:
            rospy.logfatal("could not get robot pose")
            return
        assert isinstance(self.goal, PoseStamped)
        delta_x = self.goal.pose.position.x - robot_pose.transform.translation.x
        delta_y = self.goal.pose.position.y - robot_pose.transform.translation.y
        delta_angle = self._get_yaw(self.goal.pose.orientation) - self._get_yaw(robot_pose.transform.rotation)
        delta_angle = math.fabs(angles.normalize_angle(delta_angle))
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        if distance <= 0.8 and delta_angle <= 1.0:
            return True
        else:
            return False

    def cmd_inference(self, event):
        if self.goal is None or self.is_done():
            return
        laser = torch.Tensor(self.laser).float()
        cmd_vel = Twist()
        goal = tf2_geometry_msgs.PoseStamped()
        goal.pose = self.goal.pose
        goal.header.stamp = rospy.Time(0)
        goal.header.frame_id = "map"
        try:
            target_pose = self.tf_buffer.transform(goal, "base_link")
            assert isinstance(target_pose, PoseStamped)
        except tf2_ros.TransformException as ex:
            rospy.logfatal(ex)
            rospy.logfatal("could not transform goal to the robot frame")
            return
        goal = (target_pose.pose.position.x, target_pose.pose.position.y, self._get_yaw(target_pose.pose.orientation))
        goal = torch.Tensor(goal).float()
        tensor = torch.concat((laser, goal))
        tensor = torch.reshape(tensor, (1, 1, len(tensor))).to(self.device)
        self.model.train(False)
        with torch.no_grad():
            predict = self.model(tensor)
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
    rospy.init_node("deep_motion_planner")
    planner = DeepMotionPlanner()
    rospy.spin()
