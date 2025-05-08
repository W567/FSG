import time
import numpy as np
from urdfpy import URDF
from typing import Tuple
from rospkg import RosPack
from trimesh.creation import box, uv_sphere

from skrobot.model.primitives import Axis
from skrobot.viewers import PyrenderViewer
from skrobot.coordinates import Coordinates
from skrobot.models.urdf import RobotModelFromURDF

from utils.bcolors import BColors

ros_package = RosPack()


def activate_jnt(joint_list, mimic_joint_list, mimic_multiplier_list, mimic_offset_list):
    """
    For each joint mimicking target, if target is not in the same path
    Set the first joint mimicking this target as active joint
    """
    mimic_targets = list(set(mimic_joint_list))
    joint_names = [joint.name for joint in joint_list]
    for target in mimic_targets:
        if target == 'None':
            continue
        else:
            if target not in joint_names:
                idx = mimic_joint_list.index(target)
                mimic_joint_list[idx] = 'None'
                tmp_mimic_multiplier = mimic_multiplier_list[idx]
                tmp_mimic_offset = mimic_offset_list[idx]
                active_jnt_name = joint_list[idx].name
                while target in mimic_joint_list:
                    idx = mimic_joint_list.index(target)
                    mimic_joint_list[idx] = active_jnt_name
                    mimic_multiplier_list[idx] *= tmp_mimic_multiplier
                    mimic_offset_list[idx] -= tmp_mimic_offset
    return mimic_joint_list, mimic_multiplier_list, mimic_offset_list


class Robot:

    def __init__(self):
        self.tip_namelist = None
        self.link_lists = None
        self.joint_lists = None
        self.palm_link = None
        self.skrobot_viewer = None
        self.skrobot_model = None
        self.init_pose_fk = None
        self.fk = None
        self.joint_names = None
        self.joints = None
        self.link_names = None
        self.links = None
        self.robot_urdf_path = None

    def init_urdf(
            self,
            robot: str,
            init_pose: dict
    ) -> None:
        urdf_path = f"{ros_package.get_path('fsg')}/assets/robot/{robot}/urdf/{robot}.urdf"
        robot = URDF.load(urdf_path)
        print(f"[URDF] Loaded robot model from {urdf_path}\n")
        self.robot_urdf_path = urdf_path

        self.links = robot.links
        self.link_names = [link.name for link in self.links]
        self.joints = robot.joints
        self.joint_names = [joint.name for joint in self.joints]
        self.fk = robot.link_fk()
        # initialize robot with configurable init_pose (urdf model for fk calculation)
        self.init_pose_fk = robot.link_fk(cfg=init_pose)

        self.skrobot_model = RobotModelFromURDF(urdf_file=urdf_path)

        # initialize robot with configurable init_pose (skrobot model for visualization)
        for joint in init_pose.keys():
            getattr(self.skrobot_model, joint).joint_angle(init_pose[joint])
        self.skrobot_viewer = PyrenderViewer(resolution=(640, 480))
        self.skrobot_viewer.add(self.skrobot_model)
        self.skrobot_viewer.show()

    @property
    def get_init_pose_fk(self) -> dict:
        return self.init_pose_fk

    @property
    def viewer(self) -> PyrenderViewer:
        return self.skrobot_viewer

    def get_palm(
            self,
            palm_name: str,
            manual: bool=False,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        if palm_name in self.link_names:
            palm_idx = self.link_names.index(palm_name)
        else:
            raise ValueError(f"Palm link '{palm_name}' not found")
        self.palm_link = self.links[palm_idx]

        link_co = self.init_pose_fk[self.palm_link]
        skrobot_co = Coordinates(link_co)
        axis = Axis.from_coords(skrobot_co, axis_radius=0.005, axis_length=0.1)
        self.skrobot_viewer.add(axis)
        self.skrobot_viewer.redraw()

        mesh_name, mesh_path, mesh_scale, mesh_pose = self.get_mesh(self.palm_link)

        palm_rotmat_orig = axis.worldrot()
        palm_rotmat = np.eye(4)
        while True:
            if not manual and "srh" in self.robot_urdf_path:
                rot_axis = 'q'
            else:
                print(f"{BColors.OKCYAN}===== Rotate frame to have X-O-Z plane parallel to palm ====={BColors.ENDC}")
                rot_axis = input(f"{BColors.OKBLUE}Which rotation axis? (x/y/z/q)\n"
                                 f">> {BColors.ENDC}")
            if rot_axis == 'q':
                palm_rotmat[:3, :3] = palm_rotmat_orig @ np.linalg.inv(axis.worldrot())
                time.sleep(0.5)
                break
            if rot_axis not in ['x', 'y', 'z']:
                print(f"{BColors.OKRED}Invalid input{BColors.ENDC}")
                continue
            rot_angle = input("Rotation angle in degree:\n"
                              ">> ")
            axis.rotate(np.deg2rad(float(rot_angle)), rot_axis)
            self.skrobot_viewer.redraw()

        return mesh_path, mesh_scale, palm_rotmat

    def get_palm_pose(self):
        return self.init_pose_fk[self.palm_link]

    def get_mesh(
            self,
            link,
            save_path: str=None
    ) -> Tuple[str, str, np.ndarray, np.ndarray]:
        if save_path is None:
            save_path = '/tmp'
        if link.visuals[0].geometry.mesh:
            mesh_path = link.visuals[0].geometry.mesh.filename
            if link.visuals[0].geometry.mesh.scale is not None:
                mesh_scale = link.visuals[0].geometry.mesh.scale
            else:
                mesh_scale = np.ones(3)
        elif link.visuals[0].geometry.box:
            size = link.visuals[0].geometry.box.size
            mesh_box = box(size, transform=link.visuals[0].origin)
            mesh_path = save_path + "/box_" + "_".join([str(s) for s in size]) + ".stl"
            mesh_box.export(mesh_path)
            mesh_scale = np.ones(3)
        elif link.visuals[0].geometry.sphere:
            size = link.visuals[0].geometry.sphere.radius
            mesh_sphere = uv_sphere(radius=size, transform=link.visuals[0].origin)
            mesh_path = save_path + "/sphere_" + str(size) + ".stl"
            mesh_sphere.export(mesh_path)
            mesh_scale = np.ones(3)
        else:
            raise NotImplementedError("Only mesh, box, and sphere geometries are implemented for robot links")

        mesh_name = mesh_path.split('/')[-1]
        mesh_pose = link.visuals[0].origin
        mesh_scale = np.append(mesh_scale, 1.0) # for creating homogeneous coordinate
        if 'package' in mesh_path:
            package_name = (mesh_path.split('//')[1]).split('/')[0]
            mesh_path = f"{ros_package.get_path(package_name)}/{mesh_path.split(package_name)[1]}"
        return mesh_name, mesh_path, mesh_scale, mesh_pose

    def get_child_joints(self, parent_name: str) -> list:
        """
        Get all child joints connected with the parent_link
        """
        child_joints = [joint for joint in self.joints if joint.parent == parent_name]
        return child_joints

    def get_joints_path(self, base_joint):
        """
        Recursively get all joints and links from base_joint to end link
        """
        link_res, joint_res = [], []
        next_link = [link for link in self.links if link.name == base_joint.child][0]
        next_joint = self.get_child_joints(next_link.name)

        link_list = [next_link,]
        joint_list = [base_joint,]
        if len(next_joint) == 0:
            joint_res.append(joint_list)
            link_res.append(link_list)
        else:
            for joint in next_joint:
                j_list, l_list = self.get_joints_path(joint)
                for j, l in zip(j_list, l_list):
                    joint_res.append(joint_list + j)
                    link_res.append(link_list + l)
        return joint_res, link_res

    def filter_paths(self):
        """
        Filter out paths that is not of fingers
        """
        filtered_link_lists, filtered_joint_lists = [], []
        for i, link_list in enumerate(self.link_lists):
            tip_idx = 0
            for link in link_list[::-1]:
                if link.name in self.tip_namelist and link.visuals:
                    tip_idx = link_list.index(link)
            if tip_idx == 0:
                continue
            filtered_link_lists.append(link_list[:tip_idx+1])
            filtered_joint_lists.append(self.joint_lists[i][:tip_idx])
        self.joint_lists = filtered_joint_lists
        self.link_lists = filtered_link_lists


    def get_joint_path(self, start_link_name: str, end_link_name: str) -> Tuple[list, list]:
        start_link_id = self.link_names.index(start_link_name)
        end_link_id = self.link_names.index(end_link_name)

        skrobot_link_path = self.skrobot_model.find_link_path(self.skrobot_model.link_list[start_link_id],
                                                              self.skrobot_model.link_list[end_link_id])
        skrobot_joint_path = self.skrobot_model.joint_list_from_link_list(skrobot_link_path)[1:]

        skrobot_link_path_names = [link.name for link in skrobot_link_path]

        skrobot_joint_path_names = [joint.name for joint in skrobot_joint_path]
        skrobot_joint_parent_names = [joint.parent_link.name for joint in skrobot_joint_path]
        skrobot_joint_child_names = [joint.child_link.name for joint in skrobot_joint_path]

        link_pairs = [[skrobot_link_path_names[i], skrobot_link_path_names[i + 1]] for i in range(len(skrobot_link_path_names) - 1)]

        k = 0
        extend_joint_path_names = []
        for link_pair in link_pairs:
            if (k < len(skrobot_joint_parent_names) and
                link_pair == [skrobot_joint_parent_names[k], skrobot_joint_child_names[k]]):
                extend_joint_path_names.append(skrobot_joint_path_names[k])
                k += 1
            else:
                child_joints = self.get_child_joints(link_pair[0])
                for child_joint in child_joints:
                    if child_joint.child == link_pair[1]:
                        extend_joint_path_names.append(child_joint.name)
                        break

        link_path_names = [link.name for link in skrobot_link_path]
        print("-------------------------------------------------------------")
        print(f"{BColors.OKYELLOW}Link path from {start_link_name} to {end_link_name}: {link_path_names}")
        print(f"Joint path from {start_link_name} to {end_link_name}: {extend_joint_path_names}{BColors.ENDC}")

        link_path = [self.links[self.link_names.index(name)] for name in link_path_names]
        joint_path = [self.joints[self.joint_names.index(name)] for name in extend_joint_path_names]
        return joint_path, link_path


    def get_jnt_attr(self, joint_list):
        """
        Get joint_type, mimic_joint, mimic_multiplier, mimic_offset of all joints on each path
        """
        joint_type_list, mimic_joint_list, mimic_multiplier_list, mimic_offset_list = [], [], [], []
        for joint in joint_list:
            if joint.joint_type == 'revolute':
                joint_type_list.append(1)
            elif joint.joint_type == 'fixed':
                joint_type_list.append(0)
            elif joint.joint_type == 'prismatic':
                joint_type_list.append(2)
            else:
                raise NotImplementedError("Only revolute, fixed, and prismatic joints are implemented")
            if joint.mimic is None:
                mimic_joint_list.append('None')
                mimic_multiplier_list.append(1.0)
                mimic_offset_list.append(0.0)
            else:
                mimic_joint_list.append(joint.mimic.joint)
                mimic_multiplier_list.append(joint.mimic.multiplier)
                mimic_offset_list.append(joint.mimic.offset)
        return joint_type_list, mimic_joint_list, mimic_multiplier_list, mimic_offset_list

    def determined_tf(self, link_list):
        """
        Calculate determined transform matrix between links from palm to tip
        """
        tf_list = []
        for i in range(len(link_list) -1):
            parent_co = self.fk[link_list[i]]
            child_co = self.fk[link_list[i+1]]
            tf = np.linalg.inv(parent_co) @ child_co
            tf_list.append(tf)
        return tf_list



