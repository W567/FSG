#!/usr/bin/env python
import os
import time
import yaml
import h5py
import numpy as np
import open3d as o3d

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from fsg.srv import GraspPose, GraspPoseRequest

import rospkg
ros_package = rospkg.RosPack()
from utils import publish_pcd


def are_quaternions_similar(q1, q2, threshold=5e-3):
    """
    Check if two quaternions are similar in rotation.

    Args:
        q1 (array-like): First quaternion.
        q2 (array-like): Second quaternion.
        threshold (float): Tolerance for similarity.

    Returns:
        bool: True if quaternions are similar, False otherwise.
    """
    # Normalize the quaternions
    q1 = np.array(q1) / np.linalg.norm(q1)
    q2 = np.array(q2) / np.linalg.norm(q2)

    # Compute the dot product
    dot_product = np.dot(q1, q2)

    # Check if the absolute value of the dot product is close to 1
    return abs(dot_product) > (1 - threshold)


def get_pose_similar_to_any(pose_list, pose2, hand_pose_list, hand_pose, threshold=5e-3) -> np.ndarray:
    """
    Check if pose2 is similar to any pose in pose_list.
    Returns:
        bool: True if q2 is similar to any quaternion in q_list, False otherwise.
    """
    pos2 = pose2[:3]
    q2 = pose2[3:]
    similar_indices = []
    for i, pose in enumerate(pose_list):
        pos1 = pose[:3]
        if np.linalg.norm(pos1 - pos2) > 0.005:
            continue
        q = pose[3:]
        if are_quaternions_similar(q, q2, threshold):
            hand_pose1 = hand_pose_list[i]
            if np.max(hand_pose1 - hand_pose) > 0.08:
                continue
            similar_indices.append(i)
    return np.array(similar_indices)


class Saver:
    def __init__(self):
        self.file = None
        self.group = None
        self.end = False
        self.pose_size = {}

        self.obj_name = None
        self.obj_topic_name = None
        self.hand_pose_stride = None
        self.fsg_req = None
        self.fsg_client = None

        self.fsg_num_try = 0
        self.fsg_total_time = 0
        self.fsg_total_pose = 0

        self.palm_obj_pcd = None
        self.palm_obj_aff_pcd = None
        self.palm_obj_pc_pub = None
        self.palm_obj_aff_pc_pub = None

        rospy.init_node('Saver', anonymous=True)
        self.robot_hand = rospy.get_param('robot_hand')
        self.hand_jnt_size   = rospy.get_param('hand_jnt_size')
        self.finger_idx_bridge = np.asarray(rospy.get_param('fin_idx_bridge'))

        # Pose difference threshold for similarity check
        self.pose_threshold = rospy.get_param('pose_threshold', 0.01)
        # How many grasp poses required for an object
        self.obj_threshold  = rospy.get_param('obj_threshold', 100)
        self.num_try_on_grasp = 0
        self.num_try_on_grasp_threshold = rospy.get_param('num_try_on_grasp_threshold', 10)

        self.filename = rospy.get_param('filename', 'grasp_tmp.h5')
        self.folder_path = f"{ros_package.get_path('fsg')}/hdf5/{self.robot_hand}/"
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        self.file_address = self.folder_path + self.filename

        self.log_file_address = self.folder_path + self.filename.split('.')[0] + '.log'
        self.log_unsolved_file_address = self.folder_path + self.filename.split('.')[0] + '_unsolved.log'

        self.ws_type = rospy.get_param('ws_type', 0) # 0: object-specific, 1: general

        self.with_aff = rospy.get_param('with_aff', False)
        self.obj_id = 0
        aff_id = rospy.get_param('aff_id', '0')
        self.aff_ext = f'_aff{aff_id}'
        obj_folder_name = rospy.get_param('obj_folder_name', 'obj')
        self.obj_path = f"{ros_package.get_path('fsg')}/assets/{obj_folder_name}/"
        self.obj_name_list = [name.split('.pcd')[0] for name in os.listdir(self.obj_path) if 'pcd' in name and 'aff' not in name]
        self.obj_name_list = sorted(self.obj_name_list)
        self.obj_max_num = len(self.obj_name_list)

        self.robot_palm_frame = 'ik_palm'
        self.robot_base_frame = 'robot_base_link'
        # Robot base frame to palm frame
        static_tf_br = tf2_ros.StaticTransformBroadcaster()
        palm_tf = TransformStamped()
        palm_tf.header.stamp = rospy.Time.now()
        palm_tf.header.frame_id = self.robot_base_frame
        palm_tf.child_frame_id = self.robot_palm_frame
        palm_tf.transform.translation.x = 0
        palm_tf.transform.translation.y = 0
        palm_tf.transform.translation.z = 0
        palm_tf.transform.rotation.x = 0.5
        palm_tf.transform.rotation.y = 0.5
        palm_tf.transform.rotation.z = 0.5
        palm_tf.transform.rotation.w = 0.5
        static_tf_br.sendTransform(palm_tf)

        self.init_services()
        self.timer = rospy.Timer(rospy.Duration(secs=0, nsecs=500000000), self.message_callback)
        rospy.loginfo("Saver initialized")

    def init_services(self):
        """
        Initialize services
        """
        # to generate grasp poses
        self.fsg_client = rospy.ServiceProxy('grasp_pose', GraspPose)
        self.fsg_client.wait_for_service()
        self.fsg_req = GraspPoseRequest()
        self.fsg_req.finger_num = rospy.get_param('finger_num')
        self.fsg_req.angle_num = rospy.get_param('finger_angle_num')
        self.fsg_req.with_aff = rospy.get_param('with_aff', False)

    def switch_obj(self):
        """
        Switch object, with point cloud updated
        """
        if self.obj_id >= self.obj_max_num:
            self.end = True
            return False

        if self.fsg_num_try > 0:
            with open(self.log_file_address, 'a') as f:
                f.write(f"{self.obj_name}\n")
                f.write(f"Averaged time for fsg (one execute): {self.fsg_total_time / self.fsg_num_try}\n")
                f.write(f"Averaged number of grasp poses for fsg (one execute): {self.fsg_total_pose / self.fsg_num_try}\n")
                if self.fsg_total_pose > 0:
                    f.write(f"Averaged time for one pose: {self.fsg_total_time / self.fsg_total_pose}\n")
                else:
                    f.write(f"Averaged time for one pose: null\n")

        self.obj_name = self.obj_name_list[self.obj_id]
        self.obj_id += 1

        self.fsg_total_pose = 0
        self.fsg_total_time = 0
        self.fsg_num_try = 0

        # set up point cloud
        self.palm_obj_pcd = o3d.io.read_point_cloud(self.obj_path + self.obj_name + '.pcd')
        self.palm_obj_aff_pcd = None
        # Set up affordance point cloud
        if self.with_aff:
            if os.path.exists(self.obj_path + self.obj_name + self.aff_ext + '.pcd'):
                self.palm_obj_aff_pcd = o3d.io.read_point_cloud(self.obj_path + self.obj_name + self.aff_ext + '.pcd')
            else:
                assert False, f"Affordance point cloud not found for object: {self.obj_name}"

        self.obj_topic_name = self.obj_name.replace('-', '_')
        self.fsg_req.obj_pc_topic = f"palm_{self.obj_topic_name}"
        self.fsg_req.obj_bottom_frame = "none"

        rospy.logwarn(f"[Saver] Switched to object: {self.obj_name}")
        if self.ws_type == 0:
            pc_path_file = ros_package.get_path('fsg') + f'/config/{self.obj_name}/pc_path.yaml'
        elif self.ws_type == 1:
            pc_path_file = ros_package.get_path('fsg') + f'/config/{self.robot_hand}/pc_path.yaml'
        elif self.ws_type == 2:
            rospy.logerr("[Saver] Invalid workspace type")
            return False
        rospy.logwarn("------------------------- pc path file -------------------------")
        rospy.logwarn(pc_path_file)
        with open(pc_path_file, 'r') as file:
            params = yaml.safe_load(file)
            rospy.set_param("ws_pc_paths", params['ws_pc_paths'])
            if 'link_pc_paths' in params.keys():
                rospy.set_param("link_pc_paths", params['link_pc_paths'])

        self.restart_publisher()
        return True

    def restart_publisher(self):
        """
        Restart publishers for point clouds
        """
        if self.timer is not None and self.timer.is_alive():
            self.timer.shutdown()

        if self.palm_obj_pc_pub is not None and self.palm_obj_pc_pub.impl is not None:
            self.palm_obj_pc_pub.unregister()
            if self.with_aff:
                self.palm_obj_aff_pc_pub.unregister()
        self.palm_obj_pc_pub = rospy.Publisher(f"/palm_{self.obj_topic_name}", PointCloud2, queue_size=1, latch=False)
        if self.with_aff:
            self.palm_obj_aff_pc_pub = rospy.Publisher(f"/palm_{self.obj_topic_name}{self.aff_ext}", PointCloud2,
                                                       queue_size=1, latch=False)

        self.timer = rospy.Timer(rospy.Duration(secs=0, nsecs=500000000), self.message_callback)

    def message_callback(self, event):
        """
        callback function for timer
        publish point clouds and tf
        """
        if self.palm_obj_pcd is not None and self.palm_obj_pc_pub is not None:
            publish_pcd(self.palm_obj_pcd, self.robot_palm_frame, self.palm_obj_pc_pub)
        if self.with_aff and self.palm_obj_aff_pcd is not None and self.palm_obj_aff_pc_pub is not None:
            publish_pcd(self.palm_obj_aff_pcd, self.robot_palm_frame, self.palm_obj_aff_pc_pub)

    def init_key_count(self):
        """
        Get the number of poses for each object in the file
        """
        file_list = [name for name in os.listdir(self.folder_path)]
        if not self.filename in file_list:
            return
        with h5py.File(self.file_address, 'r') as f:
            for obj_name in f.keys():
                self.pose_size[obj_name] = f[obj_name + '/palm_pose'].shape[0]

    def add_obj_key(self):
        """
        Add object key to the file
        """
        if not self.obj_name in self.file.keys():
            self.group = self.file.create_group(self.obj_name)
            self.group.create_dataset("palm_pose", shape=(1, 7),
                                      compression="gzip", chunks=True, maxshape=(None, None))
            self.group.create_dataset("hand_pose", shape=(1, self.hand_jnt_size),
                                      compression="gzip", chunks=True, maxshape=(None, None))
            self.group.create_dataset("gws", shape=(1, 1),
                                      compression="gzip", chunks=True, maxshape=(None, None))
            self.pose_size[self.obj_name] = 0

    def is_pose_enough(self):
        """
        Check if the number of poses for the current object is enough
        """
        if self.obj_name not in self.pose_size.keys():
            return False
        elif self.pose_size[self.obj_name] >= self.obj_threshold:
            return self.switch_obj()
        else:
            return False

    def is_try_enough(self):
        """
        Check if the number of tries for the current object is enough
        """
        if self.num_try_on_grasp >= self.num_try_on_grasp_threshold:
            self.num_try_on_grasp = 0
            self.log_unsolved()
            self.switch_obj()

    def log_unsolved(self):
        with open(self.log_unsolved_file_address, 'a') as f:
            f.write(self.obj_name + '\n')

    def extend_file_data(self):
        self.file[self.obj_name + '/palm_pose'].resize(self.pose_size[self.obj_name], axis=0)
        self.file[self.obj_name + '/hand_pose'].resize(self.pose_size[self.obj_name], axis=0)
        self.file[self.obj_name + '/gws'].resize(self.pose_size[self.obj_name], axis=0)

    def is_pose_similar(self, palm_pose, hand_pose, pose_threshold=0.1):
        """
        :param palm_pose: palm pose
        :param hand_pose: hand pose
        :param pose_threshold: robot pose difference threshold
        :return: True if the pose is similar to the existing poses, False otherwise
        """
        if self.pose_size[self.obj_name] == 0:
            return False
        palm_pose_list = self.file[self.obj_name + '/palm_pose']
        hand_pose_list = self.file[self.obj_name + '/hand_pose']

        found_similar_indices = get_pose_similar_to_any(palm_pose_list, palm_pose, hand_pose_list, hand_pose)
        if len(found_similar_indices) == 0:
            return False
        else:
            hand_poses = np.array(self.file[self.obj_name + '/hand_pose'])[found_similar_indices]
            hand_pose_diff = hand_poses - hand_pose
            hand_pose_errors = np.max(np.fabs(hand_pose_diff), axis=1)
            if all(e > pose_threshold for e in hand_pose_errors):
                return False
            else:
                return True

    def convert_hand_pose(self, hand_pose):
        if '85' not in self.robot_hand:
            hand_pose = np.deg2rad(np.array(hand_pose))[self.finger_idx_bridge]
        return hand_pose

    def add_pose(self, palm_pose, hand_pose, score):
        """
        Add pose to the file if it's valid
        :param palm_pose: palm pose
        :param hand_pose: hand joint angles
        :param score: grasp quality score
        :return: True if the pose is added, False otherwise
        """
        self.file = h5py.File(self.file_address, 'a')
        self.add_obj_key()

        if self.pose_size[self.obj_name] == 1 and self.file[self.obj_name + '/gws'][0] == 0.0:
            self.pose_size[self.obj_name] = 0
            self.extend_file_data()

        if self.is_pose_similar(palm_pose, hand_pose):
            self.file.close()
            return False
        else:
            self.pose_size[self.obj_name] += 1
            self.extend_file_data()

            self.file[self.obj_name + '/palm_pose'][-1] = palm_pose
            self.file[self.obj_name + '/hand_pose'][-1] = hand_pose
            self.file[self.obj_name + '/gws'][-1] = score

            rospy.loginfo(f"[Saver] Written {self.obj_name}: {self.pose_size[self.obj_name]} / {self.obj_threshold}")
            self.file.close()
            return True

    def execute(self):
        execution_start_time = time.time()

        self.switch_obj()
        self.init_key_count()
        while not rospy.is_shutdown():
            while self.is_pose_enough():
                pass
            self.is_try_enough()
            if self.end:
                rospy.loginfo("[Saver] Done")
                break

            fsg_res = self.fsg_client(self.fsg_req)
            self.fsg_total_time += fsg_res.time
            self.fsg_total_pose += fsg_res.num
            self.fsg_num_try += 1
            if fsg_res.num == 0:
                self.num_try_on_grasp += 1
                continue
            if self.hand_pose_stride is None:
                self.hand_pose_stride = fsg_res.angles.layout.dim[0].stride

            palm_poses = fsg_res.poses.poses
            new_pose_added = False
            for i, palm_pose in enumerate(palm_poses):
                hand_pose = fsg_res.angles.data[i*self.hand_pose_stride:(i+1)*self.hand_pose_stride]
                hand_pose = self.convert_hand_pose(hand_pose)
                numpy_palm_pose = np.array([palm_pose.position.x, palm_pose.position.y, palm_pose.position.z,
                                            palm_pose.orientation.w, palm_pose.orientation.x, palm_pose.orientation.y, palm_pose.orientation.z])
                pose_added = self.add_pose(numpy_palm_pose, hand_pose, fsg_res.scores[i])
                new_pose_added = new_pose_added or pose_added
                if self.is_pose_enough():
                    break
            if not new_pose_added:
                self.num_try_on_grasp += 1

        rospy.loginfo("[Saver] Total execution time: %.4f" % (time.time() - execution_start_time))

if __name__ == "__main__":
    saver = Saver()
    saver.execute()

