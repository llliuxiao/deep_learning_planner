# ROS
import rospy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# import tf2_geometry_msgs

# utils
import random
import math
import json
import sys

sys.path.append("/usr/lib/python3/dist-packages")


def get_yaw(quaternion: Quaternion):
    _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    return yaw


def get_distance_from_path(path: Path):
    distance = 0.0
    for i in range(len(path.poses) - 1):
        pose1 = path.poses[i]
        pose2 = path.poses[i + 1]
        dx = pose2.pose.position.x - pose1.pose.position.x
        dy = pose2.pose.position.y - pose1.pose.position.y
        distance += math.sqrt(dx * dx + dy * dy)
    return distance


class PoseUtils:
    def __init__(self, robot_radius, scene=None):
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer)
        self.robot_radius = robot_radius
        if scene is not None:
            with open(f"/home/gr-agv-lx91/isaac_sim_ws/src/deep_learning_planner/points/{scene}.json", "r") as f:
                self.points = json.load(f)["points"]

    def get_robot_pose(self, source_frame: str, target_frame: str):
        try:
            pose = self.buffer.lookup_transform(target_frame=target_frame,
                                                source_frame=source_frame,
                                                time=rospy.Time(0))
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = pose.transform.translation.x
            pose_stamped.pose.position.y = pose.transform.translation.y
            pose_stamped.pose.position.z = pose.transform.translation.z
            pose_stamped.pose.orientation = pose.transform.rotation
            pose_stamped.header = pose.header
            return pose_stamped
        except TransformException as ex:
            rospy.logfatal(ex)
            rospy.logfatal(f"Could not look transform from {source_frame} to {target_frame}")
            return None

    # def transform_pose(self, target_pose: PoseStamped, target_frame):
    #     try:
    #         pose = tf2_geometry_msgs.PoseStamped()
    #         pose.header = target_pose.header
    #         pose.header.stamp = rospy.Time.now()
    #         pose.pose = target_pose.pose
    #         transformed_pose = self.buffer.transform(pose, target_frame, timeout=rospy.Duration(secs=1))
    #         assert isinstance(transformed_pose, PoseStamped)
    #         return transformed_pose
    #     except TransformException as ex:
    #         rospy.logfatal(ex)
    #         rospy.logfatal(f"Could not look transform from {target_pose.header.frame_id} to {target_frame}")
    #         return None

    def get_random_pose(self, static_map: OccupancyGrid, source_frame, target_frame):
        map_width = static_map.info.width * static_map.info.resolution + static_map.info.origin.position.x
        map_height = static_map.info.height * static_map.info.resolution + static_map.info.origin.position.y
        x = random.uniform(static_map.info.origin.position.x, map_width)
        y = random.uniform(static_map.info.origin.position.y, map_height)
        while not self._is_pose_valid(x, y, static_map):
            x = random.uniform(static_map.info.origin.position.x, map_width)
            y = random.uniform(static_map.info.origin.position.y, map_height)
        theta = random.uniform(-math.pi, math.pi)
        pose = PoseStamped()
        pose.header.frame_id = source_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        (x_, y_, z_, w_) = quaternion_from_euler(0, 0, theta)
        pose.pose.orientation.x = x_
        pose.pose.orientation.y = y_
        pose.pose.orientation.z = z_
        pose.pose.orientation.w = w_
        if source_frame == target_frame:
            return pose
        else:
            return self.transform_pose(pose, target_frame)

    def get_preset_pose(self, frame):
        index = random.randint(0, len(self.points) - 1)
        (x0, y0, yaw0, x1, y1, yaw1) = self.points[index]
        robot_pose = self._make_pose(x0, y0, yaw0, frame)
        target_pose = self._make_pose(x1, y1, yaw1, frame)
        return robot_pose, target_pose

    @staticmethod
    def _make_pose(x, y, yaw, frame):
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        (x_, y_, z_, w_) = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = x_
        pose.pose.orientation.y = y_
        pose.pose.orientation.z = z_
        pose.pose.orientation.w = w_
        return pose

    def _is_pose_valid(self, x, y, static_map):
        cell_radius = int(self.robot_radius / static_map.info.resolution)
        y_index = int((y - static_map.info.origin.position.y) / static_map.info.resolution)
        x_index = int((x - static_map.info.origin.position.x) / static_map.info.resolution)
        for i in range(x_index - cell_radius, x_index + cell_radius):
            for j in range(y_index - cell_radius, y_index + cell_radius):
                index = j * static_map.info.width + i
                if index >= len(static_map.data):
                    return False
                try:
                    val = static_map.data[index]
                except IndexError:
                    print(f"IndexError: index: {index}, map_length: {len(static_map.data)}")
                    return False
                if val != 0:
                    return False
        return True
