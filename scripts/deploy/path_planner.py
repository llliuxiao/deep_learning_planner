import rospy
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from src.deep_learning_planner.scripts.sequence import SequenceModel
import torch
import numpy as np
from src.deep_learning_planner.scripts.utils.parameters import *
from src.deep_learning_planner.scripts.utils.pose_utils import get_yaw


class PathPlanner:
    def __init__(self):
        self.robot_frame = "base_link"
        self.laser_pool = []
        self.goal = None
        self.laser_pool_capacity = interval * laser_length
        self.device = torch.device("cuda:0")
        self.model = SequenceModel()
        self.model = torch.nn.DataParallel(self.model).to(device=self.device)
        self.model.eval()
        param_file = "/home/gr-agv-lx91/isaac_sim_ws/src/deep_learning_planner/transformer_logs/model11/best.pth"
        params = torch.load(param_file)["model_state_dict"]
        self.model.load_state_dict(params)
        self.scale = torch.tensor([look_ahead_distance, look_ahead_distance], dtype=torch.float)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        self.goal_sub = rospy.Subscriber("/move_base/current_goal", PoseStamped, self.goal_callback)
        self.path_pub = rospy.Publisher("/local_path", Path, queue_size=1)

        rospy.Timer(rospy.Duration(secs=0, nsecs=20000000), self.inference)

    def inference(self, event):
        if self.goal is None or len(self.laser_pool) <= 0:
            return
        laser, goal = self.make_tensor()
        predict = self.model(laser, goal).cpu()
        predict = torch.squeeze(predict)
        predict = torch.reshape(predict, shape=(-1, 2))
        predict = torch.mul(predict, self.scale)
        path = Path()
        for i in range(len(predict)):
            x, y = predict[i, 0].item(), predict[i, 1].item()
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            pose.header.frame_id = self.robot_frame
            pose.header.stamp = rospy.Time.now()
            path.poses.append(pose)
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.robot_frame
        self.path_pub.publish(path)

    def make_tensor(self):
        lasers = [self.laser_pool[-1] for _ in range(laser_length)]
        for i in range(laser_length - 1, 0, -1):
            prefix = len(self.laser_pool) - i * interval
            if prefix < 0:
                lasers[laser_length - i - 1] = self.laser_pool[0]
            else:
                lasers[laser_length - i - 1] = self.laser_pool[prefix]
        laser_tensor = torch.tensor(np.array(lasers), dtype=torch.float)
        laser_tensor = torch.div(laser_tensor, torch.tensor(laser_range, dtype=torch.float))

        # goal
        goal = tf2_geometry_msgs.PoseStamped()
        goal.pose = self.goal.pose
        goal.header.stamp = rospy.Time(0)
        goal.header.frame_id = "map"
        try:
            target_pose = self.tf_buffer.transform(goal, self.robot_frame)
            assert isinstance(target_pose, PoseStamped)
        except tf2_ros.TransformException as ex:
            rospy.logfatal(ex)
            rospy.logfatal("could not transform goal to the robot frame")
            return
        goal = (target_pose.pose.position.x, target_pose.pose.position.y, get_yaw(target_pose.pose.orientation))
        goal_tensor = torch.tensor(goal[:2], dtype=torch.float)
        goal_tensor = torch.div(goal_tensor, self.scale)
        return (torch.unsqueeze(laser_tensor, 0).to(self.device),
                torch.unsqueeze(goal_tensor, 0).to(self.device))

    def laser_callback(self, msg: LaserScan):
        if len(self.laser_pool) > self.laser_pool_capacity:
            self.laser_pool.pop(0)
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        self.laser_pool.append(ranges)

    def goal_callback(self, msg: PoseStamped):
        self.goal = msg


if __name__ == "__main__":
    rospy.init_node("local_path_planner")
    planner = PathPlanner()
    rospy.spin()
