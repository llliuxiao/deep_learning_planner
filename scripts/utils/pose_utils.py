# ROS
import rospy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# utils
import random
import math
import json
import angles


def get_yaw(quaternion: Quaternion):
    _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    return yaw


def get_pitch(quaternion: Quaternion):
    _, pitch, _ = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    return pitch


def get_roll(quaternion: Quaternion):
    roll, _, _ = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    return roll


def get_distance_from_path(path: Path):
    distance = 0.0
    for i in range(len(path.poses) - 1):
        pose1 = path.poses[i]
        pose2 = path.poses[i + 1]
        dx = pose2.pose.position.x - pose1.pose.position.x
        dy = pose2.pose.position.y - pose1.pose.position.y
        distance += math.sqrt(dx * dx + dy * dy)
    return distance


def calculate_geodesic_distance(path):
    distance = 0
    for i in range(len(path) - 1):
        dx = path[i + 1, 0] - path[i, 0]
        dy = path[i + 1, 1] - path[i, 1]
        distance += math.sqrt(dx * dx + dy * dy)
    return distance


def distance_points2d(point1, point2):
    assert hasattr(point1, "x") and hasattr(point1, "y")
    assert hasattr(point2, "x") and hasattr(point2, "y")
    delta_x = point2.x - point1.x
    delta_y = point2.y - point1.y
    return math.sqrt(delta_x ** 2 + delta_y ** 2)


def calculate_robot_position(target: tuple, relevance: tuple):
    target_x, target_y, target_yaw = target
    relevance_x, relevance_y, relevance_yaw = relevance
    robot_yaw = angles.normalize_angle(target_yaw - relevance_yaw)
    sin_yaw, cos_yaw = -math.sin(robot_yaw), math.cos(robot_yaw)
    dx = cos_yaw * relevance_x + sin_yaw * relevance_y
    dy = -sin_yaw * relevance_x + cos_yaw * relevance_y
    robot_x = target_x - dx
    robot_y = target_y - dy
    return robot_x, robot_y, robot_yaw


def calculate_relevant_position(robot_pose, target_pose):
    robot_x, robot_y, robot_yaw = robot_pose
    target_x, target_y, target_yaw = target_pose
    dx = target_x - robot_x
    dy = target_y - robot_y
    sin_yaw = -math.sin(robot_yaw)
    cos_yaw = math.cos(robot_yaw)
    x = dx * cos_yaw - dy * sin_yaw
    y = dx * sin_yaw + dy * cos_yaw
    yaw = angles.normalize_angle(target_yaw - robot_yaw)
    return x, y, yaw


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

    def transform_pose(self, target_pose: PoseStamped, target_frame):
        import tf2_geometry_msgs
        try:
            pose = tf2_geometry_msgs.PoseStamped()
            pose.header = target_pose.header
            pose.header.stamp = rospy.Time.now()
            pose.pose = target_pose.pose
            transformed_pose = self.buffer.transform(pose, target_frame, timeout=rospy.Duration(secs=1))
            assert isinstance(transformed_pose, PoseStamped)
            return transformed_pose
        except TransformException as ex:
            rospy.logfatal(ex)
            rospy.logfatal(f"Could not look transform from {target_pose.header.frame_id} to {target_frame}")
            return None

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
