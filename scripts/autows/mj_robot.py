import numpy as np
import mujoco as mj
from pathlib import Path
from copy import deepcopy

from numpy import ndarray
from scipy.spatial.transform import Rotation as R

from utils.bcolors import BColors

import rospkg
ros_package = rospkg.RosPack()


class MjRobot:

    def __init__(
            self,
            robot: str,
            tip_body_names: list,
            tip_mesh_names: list,
            obj_body_name: str
    ):
        model_path = f"{ros_package.get_path('fsg')}/assets/robot/{robot}/xml/scene.xml"
        self.asset_path = f"{ros_package.get_path('fsg')}/{robot}/xml/assets/{robot}"
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        tip_body_ids = [self.model.body(name).id for name in tip_body_names]
        geom2body_ids = [self.model.geom(i).bodyid for i in range(self.model.ngeom)]
        self.tip_mesh_name_dict = dict(zip(tip_body_ids, tip_mesh_names))
        self.tip_id_name_dict = dict(zip(tip_body_ids, tip_body_names))

        # Get all geoms under each tip body
        self.tip_geom_ids = {}
        for tip_body_id in tip_body_ids:
            current_tip_geom_ids = []
            current_tip_geom_ids.extend([i for i, body_id in enumerate(geom2body_ids) if body_id == tip_body_id])
            # Get only contact geoms under each tip body
            current_tip_contact_geom_ids = [i for i in current_tip_geom_ids if self.model.geom(i).contype != 0]
            self.tip_geom_ids[tip_body_id] = current_tip_contact_geom_ids

        # Get contact geoms under the object body
        self.obj_body_id = self.model.body(obj_body_name).id
        obj_geom_ids = [i for i, body_id in enumerate(geom2body_ids) if body_id == self.obj_body_id]
        self.obj_geom_contact_ids = [i for i in obj_geom_ids if self.model.geom(i).contype != 0]

        robot_joints_mask = np.where(self.model.jnt_type > 1)[0]
        self.robot_qposadr = self.model.jnt_qposadr[robot_joints_mask]
        robot_jnt_id = np.arange(len(self.robot_qposadr))
        self.robot_jnt_adr2id = dict(zip(self.robot_qposadr, robot_jnt_id))
        self.robot_jnt_names = [self.model.joint(i).name for i in range(self.model.njnt)]


    def get_asset_path(self):
        return self.asset_path


    def reset_keyframe(self, keyframe: int):
        mj.mj_resetDataKeyframe(self.model, self.data, keyframe)
        mj.mj_step(self.model, self.data)
        print(f"{BColors.OKYELLOW}Detected {self.data.ncon} contact points{BColors.ENDC}")


    def get_contact_points(
            self,
            tip_body_name: str,
            scale: ndarray=None,
            vis: bool=False
    ):
        contact_points = []
        contact = self.data.contact
        contact_geom_pairs = self.data.contact.geom
        found_tip_body_id = -1
        for tip_body_id, tip_geom_id_list in self.tip_geom_ids.items():
            # TODO better way required, or urdf tip body name should always be same as mj tip body name
            if tip_body_name != self.model.body(tip_body_id).name:
                continue
            for tip_geom_id in tip_geom_id_list:
                for obj_geom_id in self.obj_geom_contact_ids:
                    for i, contact_geom_pair in enumerate(contact_geom_pairs):
                        if tip_geom_id in contact_geom_pair and obj_geom_id in contact_geom_pair:
                            contact_pos = contact.pos[i]
                            # frame = contact.frame[i]

                            tip_body_pos = self.data.xpos[tip_body_id]
                            tip_body_mat = self.data.xmat[tip_body_id]

                            geom2body_pos = self.model.geom_pos[tip_geom_id]
                            geom2body_quat = self.model.geom_quat[tip_geom_id]
                            geom2body_trans = np.eye(4)
                            quat = np.array([geom2body_quat[1], geom2body_quat[2], geom2body_quat[3], geom2body_quat[0]])
                            geom2body_trans[:3, :3] = R.from_quat(quat).as_matrix()
                            geom2body_trans[:3, 3] = geom2body_pos
                            
                            geom_mesh_id = self.model.geom_dataid[tip_geom_id]
                            mesh2geom_trans = np.eye(4)
                            if geom_mesh_id != -1:
                                mesh2geom_pos = self.model.mesh_pos[geom_mesh_id]
                                mesh2geom_quat = self.model.mesh_quat[geom_mesh_id]
                                quat = np.array([mesh2geom_quat[1], mesh2geom_quat[2], mesh2geom_quat[3], mesh2geom_quat[0]])
                                mesh2geom_trans[:3, :3] = R.from_quat(quat).as_matrix()
                                mesh2geom_trans[:3, 3] = mesh2geom_pos

                            tip_body_trans = np.eye(4)
                            tip_body_trans[:3, :3] = tip_body_mat.reshape(3, 3)
                            tip_body_trans[:3, 3] = tip_body_pos

                            tip_body_trans = tip_body_trans @ geom2body_trans @ np.linalg.inv(mesh2geom_trans)

                            contact2tip = np.linalg.inv(tip_body_trans) @ np.hstack([contact_pos, 1])

                            print(f"{BColors.OKYELLOW}================================")
                            print(f"Contact {i} detected between {self.model.body(tip_body_id).name} and {self.model.body(self.obj_body_id).name}.")
                            print(f"Contact position in tip body frame: {contact2tip[:3]}{BColors.ENDC}")

                            contact_points.append(contact2tip[:3])
            if len(contact_points) == 0:
                print(f"{BColors.OKRED}No contact points found for {tip_body_name}{BColors.ENDC}")
                return None
            found_tip_body_id = tip_body_id
            break

        if vis:
            import open3d as o3d
            vis_list = []
            mesh_path = Path(self.asset_path) / self.tip_mesh_name_dict[found_tip_body_id]
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if scale is None:
                scale = [1.0, 1.0, 1.0, 1.0]
            scale_matrix = np.diag(scale)
            mesh.transform(scale_matrix)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            vis_list.append(mesh)
            vis_list.append(axis)
            for contact_point_pos in contact_points:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
                sphere.paint_uniform_color([1, 0, 0])
                sphere.translate(contact_point_pos)
                vis_list.append(deepcopy(sphere))
            if len(contact_points) > 1:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
                sphere.paint_uniform_color([0, 1, 0])
                sphere.translate(np.mean(contact_points, axis=0))
                vis_list.append(deepcopy(sphere))
            o3d.visualization.draw_geometries(vis_list)
        return np.array(contact_points)

    def get_jnt(self, tip_body_name, jnt_len):
        robot_jnt_angles = self.data.qpos[self.robot_qposadr]
        tip_dofadr = self.model.body(tip_body_name).dofadr + 1
        tip_jnt_id = self.robot_jnt_adr2id[tip_dofadr[0]]
        jnt_angles = robot_jnt_angles[int(tip_jnt_id - jnt_len + 1):int(tip_jnt_id + 1)]
        return jnt_angles


    def get_jnt_from_name(self, joint_name):
        jnt_id = self.robot_jnt_names.index(joint_name)
        return self.data.qpos[self.model.joint(jnt_id).qposadr]


    def get_jnt_all(self):
        return self.data.qpos[self.robot_qposadr]

