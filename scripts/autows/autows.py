import argparse
import os
import time
import open3d as o3d
import numpy as np
from  typing import Tuple

from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from sampler import Sampler

from robot import Robot, activate_jnt
from utils.bcolors import BColors
from utils.utils import check_duplication, get_joint_angle_combination, get_variables, fk_calc, remove_similar_rows
from utils.mesh_utils import get_mesh_size, get_palm_bounds, get_tip_interval, input_direction
from utils.yaml_utils import init_config, get_config_value, get_config_angle_range_list, ik_palm, generate_yaml
from utils.pc_utils import vis_result, pc_no_hidden, add_attribute, get_max_cluster, get_query_point, get_point_info

from mj_robot import MjRobot

import rospkg
ros_package = rospkg.RosPack()


def parser():
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument("--robot", type=str, required=True, help="robot name")
    local_parser.add_argument("--folder", type=str, default='', help="output folder name")
    local_parser.add_argument("--mj_obj_body_name", type=str, default="object", help="object body name in mujoco")
    local_parser.add_argument("--density", type=int, default=2e6, help="point density for sampling (n/m^2)")
    local_parser.add_argument("--depth_threshold", type=float, default=0.003, help="depth threshold for point selection")
    local_parser.add_argument("--normal_threshold", type=float, default=0.5, help="normal threshold for point selection")
    local_parser.add_argument("--shape", type=str, default="circle", help="shape of the point cloud [rect/circle/point]")
    local_parser.add_argument("--aug", action="store_true", help="augmentation mode")
    local_parser.add_argument("--debug", action="store_true", help="debug mode")
    local_parser.add_argument("--separate", action="store_true", help="get point cloud for each finger even with same mesh")
    local_parser.add_argument("--key", type=int, default=0, help="keyframe for grasp simulation")
    local_parser.add_argument("--manual", action="store_true", help="manual mode for contact face selection")
    local_parser.add_argument("--manual_dir", action="store_true", help="manual mode for direction selection")
    return local_parser
args = parser().parse_args()


def generate(pc_name, joint_type_list, fk_func, comb, palm_rotmat, mesh_pose, jnt_len, output_path, tf_list, x_len_list, z_len_list, shape, ext="_orig.pcd"):
    filename = f"{output_path}{pc_name}{ext}"
    with open(filename, "w") as f:

        num_rect = 0
        for tf, x_len, z_len in zip(tf_list, x_len_list, z_len_list):
            for data in comb:
                angle_list = [
                    np.deg2rad(d) if joint_type == 1
                    else d / 1000.0
                    for d, joint_type in zip(data, joint_type_list)
                ]
                arg = tuple(angle_list)

                res = fk_func(*arg)
                res = palm_rotmat @ res @ mesh_pose @ tf
                while len(data) < jnt_len:
                    data = data + (0,)
                if shape == 'rect':
                    line = ' '.join([str(x) for x in res[:3, 3]]) + ' ' + \
                           ' '.join([str(x) for x in res[:3, 1]]) + ' ' + str(0) + ' ' + \
                           ' '.join([str(x * x_len) for x in res[:3, 0]]) + ' ' + \
                           ' '.join([str(x * z_len) for x in res[:3, 2]]) + ' ' + \
                           ' '.join([str(x) for x in data]) + '\n'
                elif shape == 'circle':
                    line = ' '.join([str(x) for x in res[:3, 3]]) + ' ' + \
                           ' '.join([str(x) for x in res[:3, 1]]) + ' ' + str(0) + ' ' + \
                           ' '.join([str(x_len)]) + ' ' + \
                           ' '.join([str(0) for _ in range(5)]) + ' ' + \
                           ' '.join([str(x) for x in data]) + '\n'
                elif shape == 'point':
                    line = ' '.join([str(x) for x in res[:3, 3]]) + ' ' + \
                           ' '.join([str(x) for x in res[:3, 1]]) + ' ' + str(0) + ' ' + \
                           ' '.join([str(0) for _ in range(6)]) + ' ' + \
                           ' '.join([str(x) for x in data]) + '\n'
                f.write(line)
            num_rect += 1
        print(f"{BColors.OKGREEN}[RESULT] Number of rectangles: {pc_name} ~ {num_rect}{BColors.ENDC}")

    add_attribute(filename, jnt_len)


def generate_link_pc(pc_namelist, preprocessed_pc_namelist, link_func_list_list, palm_rotmat, max_jnt_len, output_path):
    d = 0.005
    link_pc_filenames = []
    link_pc_paths = []
    i = 0
    for pc_name in pc_namelist:
        if pc_name is None:
            continue
        preprocessed_pc_name = preprocessed_pc_namelist[i]

        link_func_list = link_func_list_list[i]
        datalist = []
        with open(preprocessed_pc_name, "r") as f:
            for line in f.readlines()[11:]:
                data = [float(x) for x in line.split(' ')[13:]]
                datalist.append(data)
        datalist = [list(item) for item in set(map(tuple, datalist))]
        filename = f"{output_path}finger_link_{i}.pcd"
        with open(filename, "w") as pcd_f:
            for data in datalist:
                all_interpolated_points = []
                prev_point = None
                for link_func in link_func_list:
                    variable = link_func[0]
                    func = link_func[1]
                    deg_data = data[:len(variable)]
                    arg = tuple([np.deg2rad(d) for d in deg_data])
                    point = func(*arg)
                    point = palm_rotmat @ point
                    point = point[:3, 3]
                    if prev_point is None:
                        prev_point = point
                        interpolated_points = [point]
                    else:
                        distance = np.linalg.norm(point - prev_point)
                        n_segments = int(np.ceil(distance / d))
                        t_values = np.linspace(0, 1, n_segments + 1)

                        interpolated_points = [prev_point + t * (point - prev_point) for t in t_values]
                        interpolated_points.append(point)
                        prev_point = point
                    all_interpolated_points.extend(interpolated_points)

                all_interpolated_points = np.array(all_interpolated_points)
                all_interpolated_points = remove_similar_rows(all_interpolated_points, threshold=d/10.0)
                for interpolated_point in all_interpolated_points:
                    line = ' '.join([str(x) for x in interpolated_point]) + ' ' + \
                           ' '.join([str(0) for _ in range(4)]) + ' ' + \
                           ' '.join([str(0) for _ in range(6)]) + ' ' + \
                           ' '.join([str(x) for x in data]) + '\n'
                    pcd_f.write(line)
        add_attribute(filename, max_jnt_len)
        link_pc_paths.append(filename)
        link_pc_filenames.append(f"finger_link_{i}")
        i += 1
    return link_pc_filenames, link_pc_paths


def get_index(pc, centroid):
    # get index list of valid points in each cloud
    idx_list = []
    points = np.asarray(pc.points)
    normals = np.asarray(pc.normals)
    for j, (point, normal) in enumerate(zip(points, normals)):
        vec = point - centroid
        vec = vec / np.linalg.norm(vec)
        # TODO threshold adjustment
        if vec.dot(normal) > 0.707:
            idx_list.append(j)
    return idx_list


def get_result(filename, tip_y_direction, jnt_len):
    # get all valid points in a cloud and write it down
    ny_threshold = tip_y_direction * -0.001
    preprocessed_filename = filename.split('_orig.pcd')[0] + '.pcd'
    with open(filename, "r") as f, open(preprocessed_filename, "w") as f_pre:
        for line in np.asarray(f.readlines()[11:]):
            if ny_threshold > 0:
                if float(line.split(' ')[4]) < ny_threshold:
                    f_pre.write(line)
            else:
                if float(line.split(' ')[4]) > ny_threshold:
                    f_pre.write(line)
    add_attribute(preprocessed_filename, jnt_len)
    return preprocessed_filename


def preprocess(pc_output_path, pc_namelist, jnt_len, tip_y_direction_list, ext="_orig.pcd"):
    preprocessed_pc_namelist = []
    for i, pc_name in enumerate(pc_namelist):
        filename = f"{pc_output_path}{pc_name}{ext}"
        new_filename = get_result(filename, tip_y_direction_list[i], jnt_len)
        preprocessed_pc_namelist.append(new_filename)
    return preprocessed_pc_namelist


class AutoWS:
    def __init__(self):
        start = time.time()

        if args.folder == '':
            folder = args.robot
        else:
            folder = args.folder
        pc_output_path = f"{ros_package.get_path('fsg')}/workspace/{folder}/"
        config_output_path = f"{ros_package.get_path('fsg')}/config/{folder}/"
        if not os.path.isdir(pc_output_path):
            os.makedirs(pc_output_path)
        if not os.path.isdir(config_output_path):
            os.makedirs(config_output_path)

        config = init_config(args.robot)
        palm_name = get_config_value(config, 'palm_name')
        tip_namelist = get_config_value(config, 'tip_namelist')
        underactuated_list = get_config_value(config, 'underactuated')
        num_step = get_config_value(config, 'num_step', default=2)
        pc_namelist = get_config_value(config, 'pc_namelist')
        angle_range_list = get_config_value(config, 'angle_range_list')
        tip_y_direction_list = get_config_value(config, 'tip_y_direction_list')

        tip_body_namelist = get_config_value(config, 'tip_body_namelist')
        tip_mesh_namelist = get_config_value(config, 'tip_mesh_namelist')

        if 'srh' not in args.robot and not args.manual_dir:
            print(f"{BColors.OKRED} Robots except for srh should manually adjust mesh direction{BColors.ENDC}"
                  f"{BColors.OKYELLOW} --manual_dir {BColors.ENDC}")
            exit()

        mesh_save_path = None
        if args.aug:
            self.mj_robot = MjRobot(args.robot, tip_body_namelist, tip_mesh_namelist, args.mj_obj_body_name)
            self.mj_robot.reset_keyframe(args.key)
            mesh_save_path = self.mj_robot.get_asset_path()

        self.robot = Robot()
        init_pose = get_config_value(config, 'init_pose')
        self.robot.init_urdf(args.robot, init_pose)

        self.pc_sampler = Sampler()

        palm_min_max, palm_rotmat = self.get_palm_info(palm_name)

        # recursively checking joints and find joint path from palm to fingertips
        # # get child joints of palm link
        # base_joints = self.robot.get_child_joints(palm_name)
        # # get joint path from palm link to the end (fingertips)
        # for base_joint in base_joints:
        #     joint_list, link_list = self.robot.get_joints_path(base_joint)
        #     for j, l in zip(joint_list, link_list):
        #         self.joint_lists.append(j)
        #         self.link_lists.append([self.robot.palm_link] + l)
        # self.filter_paths()

        mesh_namelist = []
        forward_list = []
        upward_list = []
        tip_interval_list = []
        pcd_list = []

        path_jnt_len_list = []
        fk_func_list = []
        joint_angle_comb_list = []
        tip_tf_lists = []
        x_len_lists = []
        z_len_lists = []

        joint_type_lists = []
        mesh_pose_list = []

        link_func_list_list = []

        for (tip_name, underactuated) in zip(tip_namelist, underactuated_list):

            joint_path, link_path = self.robot.get_joint_path(palm_name, tip_name)
            type_list, mimic_list, mimic_multiplier_list, mimic_offset_list = self.robot.get_jnt_attr(joint_path)
            mimic_list, mimic_multiplier_list, mimic_offset_list = activate_jnt(joint_path, mimic_list, mimic_multiplier_list, mimic_offset_list)
            var_list, min_list, max_list = get_variables(joint_path, type_list, mimic_list)
            path_jnt_len = len(set(var_list))
            if 'fixed' in set(var_list):
                path_jnt_len -= 1

            det_tf = self.robot.determined_tf(link_path)

            tip_link = link_path[-1]
            mesh_name, mesh_path, mesh_scale, mesh_pose = self.robot.get_mesh(tip_link, save_path=mesh_save_path)

            if args.aug:
                tip_body_name = tip_body_namelist[tip_namelist.index(tip_name)]
            else:
                tip_body_name = None

            mesh_index = check_duplication(mesh_name, mesh_namelist)
            if mesh_index is not None and not args.separate:
                forward = forward_list[mesh_index]
                upward = upward_list[mesh_index]
                tip_interval = tip_interval_list[mesh_index]
                full_pcd = pcd_list[mesh_index]

                tip_tf_list = tip_tf_lists[mesh_index]
                x_len_list = x_len_lists[mesh_index]
                z_len_list = z_len_lists[mesh_index]
            else:
                if mesh_index is None:
                    mesh_center, mesh_extents, mesh_bounds = get_mesh_size(mesh_path, mesh_scale)
                    forward, upward = self.get_mesh_direction(tip_link, mesh_extents, mesh_pose)
                    tip_interval = get_tip_interval(upward, forward, mesh_extents)
                    # if not exist
                    path = os.path.dirname(os.path.abspath(__file__)) + '/tmp/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    files = os.listdir(path)
                    tip_pc_name = mesh_name.split('.')[0] + '.pcd'
                    if tip_pc_name in files:
                        full_pcd = o3d.io.read_point_cloud(path + tip_pc_name)
                    else:
                        pcd = self.pc_sampler.sample(mesh_path, mesh_scale, args.density)
                        pcd = pc_no_hidden(pcd, mesh_center, mesh_extents, forward, upward, args.debug)
                        full_pcd = get_max_cluster(pcd, args.debug)
                        o3d.io.write_point_cloud(path + tip_pc_name, full_pcd)
                else:
                    forward = forward_list[mesh_index]
                    upward = upward_list[mesh_index]
                    tip_interval = tip_interval_list[mesh_index]
                    full_pcd = pcd_list[mesh_index]

                if args.aug:
                    # contact point from grasp simulation
                    contact_points = self.mj_robot.get_contact_points(tip_name, mesh_scale, args.debug)
                    if contact_points is None:
                        joint_type_lists.append(None)
                        fk_func_list.append(None)
                        joint_angle_comb_list.append(None)
                        mesh_pose_list.append(None)
                        tip_tf_lists.append(None)
                        x_len_lists.append(None)
                        z_len_lists.append(None)
                        pc_namelist[tip_namelist.index(tip_name)] = None
                        continue
                    elif len(contact_points.shape) == 1:
                        contact_point = contact_points
                    elif len(contact_points.shape) == 2:
                        contact_point = np.mean(contact_points, axis=0)
                    else:
                        raise ValueError(f"Invalid contact points")
                else:
                    # no contact point provided
                    contact_point = None

                query_point = get_query_point(full_pcd, forward, input_query=contact_point, vis=args.debug)
                tf, x_len, z_len, query_points = get_point_info(full_pcd, query_point, upward, args.normal_threshold, args.depth_threshold, args.shape, args.debug)

                if tf is None:
                    raise ValueError(f"{BColors.OKRED}Initial query point invalid, exit.{BColors.ENDC}")

                tip_tf_list, x_len_list, z_len_list = [tf], [x_len], [z_len]
                tree = o3d.geometry.KDTreeFlann(full_pcd)
                for q in query_points:
                    [_, idx, dist] = tree.search_knn_vector_3d(q, 1)
                    if dist[0] > 2.5e-5:
                        continue
                    query = full_pcd.select_by_index(idx)
                    tf, x_len, z_len, unused_query_points = get_point_info(full_pcd, query, upward, args.normal_threshold, args.depth_threshold, args.shape, args.debug, args.manual)
                    if args.manual:
                        if tf is not None:
                            tip_tf_list.append(tf)
                            x_len_list.append(x_len)
                            z_len_list.append(z_len)
                    else:
                        if x_len > 0.5 * x_len_list[0] and z_len > 0.5 * z_len_list[0]:
                            tip_tf_list.append(tf)
                            x_len_list.append(x_len)
                            z_len_list.append(z_len)

            fk_func, cal_var_list, link_func_list = fk_calc(joint_path, det_tf, type_list, mimic_multiplier_list, mimic_offset_list, var_list)

            config_min_list, config_max_list, config_step_list = get_config_angle_range_list(joint_path, angle_range_list, type_list)
            config_step_list = np.array(config_step_list)
            if args.aug:
                if path_jnt_len == 1:
                    # TODO for robotiq85 (has to be verified)
                    # idx = var_list.index(cal_var_list[0])
                    jnt_name = joint_path[0].name
                    jnt_angles = self.mj_robot.get_jnt_from_name(jnt_name)
                else:
                    jnt_angles = self.mj_robot.get_jnt(tip_body_name, path_jnt_len)
                jnt_angles = np.rad2deg(jnt_angles)
                
                config_step_list = np.ones_like(config_step_list) * 5

                config_min_list = jnt_angles - num_step * config_step_list
                config_max_list = jnt_angles + num_step * config_step_list

            min_list = np.maximum(min_list, config_min_list)
            max_list = np.minimum(max_list, config_max_list)

            cal_var_list_indices = []
            for cal_var in cal_var_list:
                idx = var_list.index(cal_var)
                cal_var_list_indices.append(idx)
            min_list = min_list[cal_var_list_indices]
            max_list = max_list[cal_var_list_indices]
            config_step_list = config_step_list[cal_var_list_indices]

            print(f"{BColors.OKYELLOW}Joint angle ranges:")
            print(f"min_list: {min_list}")
            print(f"max_list: {max_list}")
            print(f"underactuated: {underactuated}")
            print(f"config_step_list: {config_step_list}{BColors.ENDC}")

            joint_angle_comb = get_joint_angle_combination(min_list, max_list, config_step_list, underactuated)

            mesh_namelist.append(mesh_name)
            forward_list.append(forward)
            upward_list.append(upward)
            tip_interval_list.append(tip_interval)
            pcd_list.append(full_pcd)
            path_jnt_len_list.append(path_jnt_len)
            fk_func_list.append(fk_func)
            joint_angle_comb_list.append(joint_angle_comb)
            tip_tf_lists.append(tip_tf_list)
            x_len_lists.append(x_len_list)
            z_len_lists.append(z_len_list)
            joint_type_lists.append(type_list)
            mesh_pose_list.append(mesh_pose)
            link_func_list_list.append(link_func_list)

        max_jnt_len = max(path_jnt_len_list)

        filtered_pc_namelist = []
        for i, pc_name in enumerate(pc_namelist):
            if pc_name is None:
                continue
            filtered_pc_namelist.append(pc_name)
            joint_type_list = joint_type_lists[i]
            fk_func = fk_func_list[i]
            comb = joint_angle_comb_list[i]
            mesh_pose = mesh_pose_list[i]
            tf_list = tip_tf_lists[i]
            x_len_list = x_len_lists[i]
            z_len_list = z_len_lists[i]
            generate(pc_name, joint_type_list, fk_func, comb, palm_rotmat, mesh_pose, max_jnt_len, pc_output_path, tf_list, x_len_list, z_len_list, args.shape, ext="_orig.pcd")

        if args.debug:
            vis_result(pc_output_path, filtered_pc_namelist, "_orig.pcd", "original point cloud")

        urdf_path = ros_package.get_path('fsg') + '/assets/robot/' + args.robot + '/urdf/' + args.robot + '.urdf'
        xacro_path = ros_package.get_path('fsg') + '/assets/robot/' + args.robot + '/urdf/' + args.robot + '.urdf.xacro'

        preprocessed_pc_namelist = preprocess(pc_output_path, filtered_pc_namelist, max_jnt_len, tip_y_direction_list, ext="_orig.pcd")
        if args.debug:
            vis_result(pc_output_path, filtered_pc_namelist, ".pcd", "preprocessed point cloud")

        link_pc_namelist, link_pc_paths = generate_link_pc(pc_namelist, preprocessed_pc_namelist, link_func_list_list, palm_rotmat, max_jnt_len, pc_output_path)
        if args.debug:
            vis_result(pc_output_path, link_pc_namelist, ".pcd", "preprocessed point cloud")

        ik_palm(palm_rotmat, palm_name, urdf_path, xacro_path)
        generate_yaml(pc_namelist, preprocessed_pc_namelist, link_pc_paths, tip_interval_list, max_jnt_len, palm_min_max, config_output_path)

        print(f"{BColors.OKGREEN}\n[TIME] Cost time: {time.time() - start}\n{BColors.ENDC}")


    def get_palm_info(self, palm_name):
        mesh_path, mesh_scale, palm_rotmat = self.robot.get_palm(palm_name, args.manual_dir)
        # get palm bounding box size (y_max is set to 1.0, TODO which may be adjusted)
        palm_min_max, palm_center, palm_extents = get_palm_bounds(mesh_path, mesh_scale, palm_rotmat)

        # visualize palm bounding box
        palm_co = self.robot.get_palm_pose()
        palm_pos = palm_co @ np.array([palm_center[0], palm_center[1], palm_center[2], 1])
        box = Box(extents=palm_extents, pos=palm_pos[:3])
        box.set_color([0.5, 0.5, 0.0, 0.7])
        self.robot.skrobot_viewer.add(box)
        self.robot.skrobot_viewer.redraw()

        return palm_min_max, palm_rotmat


    def get_mesh_direction(self, link, mesh_extents, mesh_pose) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get forward and upward direction under tip coordinate system (original mesh coordinate system)
        """
        axes_size = np.max(mesh_extents) * 1.5

        link_co = self.robot.get_init_pose_fk[link]
        skrobot_co = Coordinates(link_co)
        axis = Axis.from_coords(skrobot_co, axis_radius=axes_size/50.0, axis_length=axes_size)
        self.robot.skrobot_viewer.add(axis)
        self.robot.skrobot_viewer.redraw()

        forward = input_direction("forward", args.manual_dir)
        upward = input_direction("upward", args.manual_dir)
        sidewards = np.cross(upward, forward)
        sidewards = sidewards / np.linalg.norm(sidewards)
        upward = np.cross(forward, sidewards)

        self.robot.skrobot_viewer.delete(axis)
        # the input forward and upward are under transformed coordinate system
        # but point cloud sampled from mesh are in the mesh original coordinate system
        inv_mesh_pose = np.linalg.inv(mesh_pose)
        forward = inv_mesh_pose[:3, :3].dot(forward)
        upward = inv_mesh_pose[:3, :3].dot(upward)
        return forward, upward


if __name__ == '__main__':
    handle = AutoWS()

