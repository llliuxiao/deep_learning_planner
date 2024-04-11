# This is a script to map high-resolution lidar data to 0.25deg

import rospy
from sensor_msgs.msg import LaserScan
import numpy as np

laser_points = 1080
angle_min = -2.356194496154785
angle_max = 2.3518311977386475
angle_increment = 0.004363323096185923
range_min = 0.4
range_max = 10.0


def map_scan(origin_laser: LaserScan):
    laser = np.ones(laser_points) * 10.0
    start = 0
    for i in range(laser_points):
        map_angle = angle_min + angle_increment * i
        min_dis = 1.0
        for j in range(start, len(origin_laser.ranges)):
            origin_angle = origin_laser.angle_min + origin_laser.angle_increment * j
            delta_angle = origin_angle - map_angle
            if delta_angle > angle_increment / 2:
                start = j
                break
            elif abs(delta_angle) < min_dis:
                min_dis = abs(delta_angle)
                laser[i] = max(range_min, min(range_max, origin_laser.ranges[j]))
    return laser


def laser_callback(msg: LaserScan):
    scan = LaserScan()
    ranges = map_scan(msg)
    scan.header.stamp = rospy.Time.now()
    scan.header.frame_id = msg.header.frame_id
    scan.angle_min = angle_min
    scan.angle_max = angle_max
    scan.angle_increment = angle_increment
    scan.time_increment = 0
    scan.scan_time = 0
    scan.range_min = range_min
    scan.range_max = range_max
    scan.ranges = ranges
    scan.intensities = [255 for _ in range(laser_points)]
    laser_pub.publish(scan)


if __name__ == "__main__":
    rospy.init_node("map_scan")
    laser_sub = rospy.Subscriber("/robot4/scan", LaserScan, laser_callback, queue_size=1)
    laser_pub = rospy.Publisher("/map_scan", LaserScan, queue_size=1)
    rospy.spin()
