import sys
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R

from skrobot.model.robot_model import RobotModel
from skrobot.coordinates.base import Coordinates
from skrobot.coordinates.math import quaternion2matrix

import rospkg
from std_msgs.msg import Header
ros_package = rospkg.RosPack()
sys.path.append(ros_package.get_path('pcl_interface') + '/scripts')
from pcBase import *


def read_pose(group_read, idx):
    hand_pose = group_read['hand_pose'][idx]
    palm_pose = group_read['palm_pose'][idx]
    if len(hand_pose.shape) == 2:
        hand_pose = hand_pose[0]
    gws = group_read['gws'][idx]
    print('GWS: %.4f' % gws)
    return hand_pose, palm_pose


def set_robot_state(
        robot_model: RobotModel,
        joint_names: List[str],
        hand_pose: np.ndarray,
        palm_pose: np.ndarray,
        robot_name: str
) -> None:
    xyz, quat = palm_pose[:3], palm_pose[3:]

    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    mat[:3, 3] = xyz

    rot_mat = np.eye(4)
    if robot_name == 'allegro':
        rot_mat[:3, :3] = R.from_euler('z', -np.pi / 2.0).as_matrix()
    elif robot_name == 'barrett':
        rot_mat[:3, :3] = R.from_euler('x', -np.pi / 2.0).as_matrix() @ R.from_euler('y', -np.pi).as_matrix()
    elif robot_name == 'robotiq85':
        rot_mat[:3, :3] = R.from_euler('x', np.pi / 2.0).as_matrix()
        hand_pose = hand_pose[0:1]
        hand_pose = np.deg2rad(hand_pose)

    mat = mat @ rot_mat
    xyz, quat = mat[:3, 3], R.from_matrix(mat[:3, :3]).as_quat()
    quat = [quat[3], quat[0], quat[1], quat[2]]

    co = Coordinates(pos=list(xyz), rot=quaternion2matrix(quat))
    robot_model.newcoords(co)

    for joint_name, angle in zip(joint_names, hand_pose):
        robot_model.__dict__[joint_name].joint_angle(angle)


def qpos_sort(data, robot_name):
    if '85' in robot_name:
        hand_qpos = np.array([1, 1, 1, -1, 1, 1, 1, -1]) * data[-1]
        data = np.concatenate((data[0:-1], hand_qpos))
    return " ".join(str(j) for j in data)

def ctrl_sort(data, robot_name, num=False):
    if 'srh' in robot_name:
        index = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29]
        data = data[index]
    elif '85' in robot_name:
        data = np.concatenate((data[0:6], [data[6] * 255 / 0.8]))

    if num:
        return data
    else:
        return " ".join(str(j) for j in data)

def hand_pos_sort(data, robot_name):
    if 'srh' in robot_name:
        index = [0,1,2,4,5,6,8,9,10,12,13,14,15,17,18,19,20,21]
        data = data[index]
    else:
        raise NotImplementedError
    return data

def quat_conjugate(a):
    return np.concatenate((-a[:3], a[-1:]))

def quat_mul(a, b):
    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w])
    return quat

def intersection_angle(a, b):
    a = np.concatenate((a[1:], a[:1])) # w,x,y,z -> x,y,z,w
    b = np.concatenate((b[1:], b[:1]))
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    diff = quat_mul(a, quat_conjugate(b))
    angle = 2.0 * np.arcsin(np.clip(np.linalg.norm(diff[0:3]), a_min=-1.0, a_max=1.0))
    return angle

def np2pc2(np_cloud, fields, frame_id=None):
    """
    :param np_cloud: point cloud in numpy array
    :param fields: point cloud fields
    :param frame_id: tf frame
    :return: point cloud message
    """
    pc_header = Header()
    pc_header.stamp = rospy.Time.now()
    if frame_id is not None:
        pc_header.frame_id = frame_id
    return pc2.create_cloud(pc_header, fields, np_cloud)

def publish_pcd(pcd, frame_id, publisher, fields=xyz_nor_fields):
    """
    :param pcd: point cloud to be published
    :param frame_id: tf frame
    :param publisher: which publisher to use
    :param fields: point cloud fields [default: xyz_nor_fields]
    """
    if pcd is not None:
        xyz = np.asarray(pcd.points)
        nor = np.asarray(pcd.normals)
        cur = np.zeros((xyz.shape[0], 1))
        publisher.publish(np2pc2(np.concatenate((xyz, nor, cur), axis=-1), fields, frame_id))
