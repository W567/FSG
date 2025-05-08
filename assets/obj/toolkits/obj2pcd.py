import os
import numpy as np
import open3d as o3d
import pc_annotation
from tqdm import tqdm
from sample_color import Sampler
from utils import extract_affordance

import argparse
parser = argparse.ArgumentParser(description="Convert a folder of .obj files")
parser.add_argument("-i", "--input_folder", help="Folder containing the .obj files")
parser.add_argument("-o","--output_folder", help="Folder to save the .pcd files")
parser.add_argument("-n", "--num_of_points", type=int, default=1024, help="Number of points to sample, [default: 1024]")
parser.add_argument("-v", "--vis", action="store_true", help="Visualize the point cloud, [default: False]")
parser.add_argument("-a", "--annotate", action="store_true", help="Annotate the point cloud to get label file, [default: False]")
parser.add_argument("-aff", "--affordance", action="store_true", help="Annotate and extract affordance point cloud, [default: False]")
parser.add_argument("-f", "--filter", type=str, default=None, help="Filter for the objects (name)")
cli_args = parser.parse_args()

co = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.2, origin = [0, 0, 0])

current_path = os.getcwd()

read_path = f"{current_path}/{cli_args.input_folder}"
save_path = f"{current_path}/{cli_args.output_folder}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

obj_list = [name for name in os.listdir(read_path) if '.obj' in name and 'collision' not in name]
obj_list.sort()

for obj in tqdm(obj_list):
    if cli_args.filter and cli_args.filter not in obj:
        continue

    obj_name = obj.split('.')[0]
    pcd_name = obj_name + ".pcd"
    pcd_path = save_path + pcd_name

    if os.path.exists(pcd_path):
        print(f"File {pcd_path} already exists, read from file...")
        pcd = o3d.io.read_point_cloud(pcd_path)
    else:
        tar_name = read_path + obj
        mesh = o3d.io.read_triangle_mesh(tar_name, True)
        if cli_args.vis:
            print(f"========= Original mesh: {obj_name} ============")
            o3d.visualization.draw_geometries([mesh, co])

        sampler = Sampler()
        pcd = sampler.sample(path=read_path + obj, num_of_points=cli_args.num_of_points, sample_color=True)
        o3d.io.write_point_cloud(save_path + pcd_name, pcd)

    if cli_args.vis:
        print(f"========= Generated cloud: {obj_name} ============")
        o3d.visualization.draw_geometries([pcd, co], point_show_normal=True)
    if cli_args.annotate:
        print(f"========= Annotation {obj_name} ============")
        pc_annotation.annotate(np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors), save_path + pcd_name)
    if cli_args.affordance:
        extract_affordance(pcd_path=save_path + pcd_name, save_path=save_path)

