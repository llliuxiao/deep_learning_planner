import json
import math
import threading
import actionlib
import rospy
from move_base_msgs.msg import MoveBaseWaypointsAction, MoveBaseWaypointsGoal, MoveBaseGoal, MoveBaseAction
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# points = [
#     (-3, 10, 0),
#     (-3, 15, 0),
#     (-3, 27, 0),
#     (-8, 28, math.pi),
#     (-15, 26, -math.pi / 2),
#     (-15, 16, -math.pi / 2),
#     (-10, 8, 0)
# ]
points = [
    (-4.85, 0.0, math.pi / 2),
    (0.0, 2.85, 0.0),
    (1.55, -3.0, -math.pi / 2),
    (0.0, -7.5, math.pi),
    (-2.4, -5.0, math.pi / 2)
]


def transform(point):
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.pose.position.x = point[0]
    goal.pose.position.y = point[1]

    angle = quaternion_from_euler(0, 0, point[2])
    goal.pose.orientation.x = angle[0]
    goal.pose.orientation.y = angle[1]
    goal.pose.orientation.z = angle[2]
    goal.pose.orientation.w = angle[3]
    return goal


def client():
    via_points = MoveBaseWaypointsGoal()
    path = Path()
    for point in points:
        via_points.target_poses.append(transform(point))
        path.poses.append(transform(point))

    via_point_client = actionlib.SimpleActionClient("/robot4/move_base_waypoints", MoveBaseWaypointsAction)
    publisher = rospy.Publisher("/via_points_path", Path, queue_size=1)
    publisher.publish(path)
    via_point_client.wait_for_server()
    via_point_client.send_goal(via_points)
    via_point_client.wait_for_result()
    result = via_point_client.get_goal_status_text()
    print(result)


if __name__ == "__main__":
    rospy.init_node("waypoints_navigation")
    thread = threading.Thread(target=client)
    thread.start()
    rospy.spin()
