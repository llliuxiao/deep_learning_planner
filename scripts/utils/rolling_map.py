import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
from nav_msgs.msg import OccupancyGrid
import math
import angles
from pose_utils import PoseUtils, get_yaw
from geometry_msgs.msg import Pose
import cv2 as cv


class RollingMap:
    def __init__(self, map_size, resolution):
        self.map_size = map_size
        self.resolution = resolution
        self.map = np.zeros(shape=(map_size, map_size), dtype=np.int8)
        self.map_pub = rospy.Publisher("/local_map", OccupancyGrid, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        self.pose_util = PoseUtils(robot_radius=0.5)
        robot_pose = self.pose_util.get_robot_pose("base_link", "map")
        while robot_pose is None:
            robot_pose = self.pose_util.get_robot_pose("base_link", "map")
            print("waiting")
            rospy.sleep(1.0)
        self.origin = (robot_pose.pose.position.x, robot_pose.pose.position.y, get_yaw(robot_pose.pose.orientation))
        self.center = np.array([map_size // 2, map_size // 2])
        self.scan = None
        rospy.Timer(period=rospy.Duration(nsecs=100_000_000), callback=self.update)

    def update(self, event):
        if self.scan is None:
            return
        robot_pose = self.pose_util.get_robot_pose("base_link", "map")
        robot_x, robot_y = robot_pose.pose.position.x, robot_pose.pose.position.y
        robot_yaw = get_yaw(robot_pose.pose.orientation)
        dx, dy = robot_x - self.origin[0], robot_y - self.origin[1]
        dx_index, dy_index = round(dx / self.resolution), round(dy / self.resolution)
        self.rolling(dx_index, dy_index)
        self.update_scan(robot_yaw)
        self.origin = (robot_x, robot_y, robot_yaw)
        self.publish_map(robot_pose.pose)

    def rolling(self, dx, dy):
        new_map = np.zeros(shape=(self.map_size, self.map_size), dtype=np.int8)
        if abs(dx) <= self.map_size and abs(dy) <= self.map_size:
            old_start_x, old_start_y = max(0, dx), max(0, dy)
            old_end_x, old_end_y = min(self.map_size, self.map_size + dx), min(self.map_size, self.map_size + dy)
            start_x, start_y = max(0, -dx), max(0, -dy)
            end_x, end_y = min(self.map_size, self.map_size - dx), min(self.map_size, self.map_size - dy)
            new_map[start_x:end_x, start_y:end_y] = self.map[old_start_x:old_end_x, old_start_y:old_end_y]
        self.map = new_map
        cv.imshow("test", new_map)
        cv.waitKey()

    def publish_map(self, robot_pose):
        ros_map = OccupancyGrid()
        ros_map.header.stamp = rospy.Time.now()
        ros_map.header.frame_id = "map"
        ros_map.info.width = self.map_size
        ros_map.info.height = self.map_size
        ros_map.info.resolution = self.resolution
        origin = Pose()
        origin.position.x = robot_pose.position.x - self.map_size / 2 * self.resolution
        origin.position.y = robot_pose.position.y - self.map_size / 2 * self.resolution
        print(f"robot_x:{robot_pose.position.x}, robot_y:{robot_pose.position.y}")
        print(f"origin_x:{origin.position.x}, origin_y:{origin.position.y}")
        ros_map.info.origin = origin
        ros_map.data = self.map.flatten(order="F")
        self.map_pub.publish(ros_map)

    def update_scan(self, robot_yaw):
        angle_min, angle_max = self.scan.angle_min, self.scan.angle_max
        angle_increment = self.scan.angle_increment
        range_max, range_min = self.scan.range_max, self.scan.range_min
        for i in range(len(self.scan.ranges)):
            point_yaw = angle_min + i * angle_increment
            distance = np.clip(self.scan.ranges[i], range_min, range_max)
            yaw = angles.normalize_angle(robot_yaw + point_yaw)
            x, y = distance * math.cos(yaw), distance * math.sin(yaw)
            x_index, y_index = round(x / self.resolution), round(y / self.resolution)
            if self.in_map(x_index, y_index):
                self.map[self.center[0] + x_index, self.center[1] + y_index] = 100

    def scan_callback(self, msg: LaserScan):
        self.scan = msg

    def in_map(self, x, y):
        return -self.map_size / 2 < x < self.map_size / 2 and -self.map_size / 2 < y < self.map_size / 2


if __name__ == "__main__":
    rospy.init_node("rolling_map")
    rolling_map = RollingMap(map_size=200, resolution=0.05)
    rospy.spin()
