import os
import numpy as np
import pc_annotation
import open3d as o3d


def extract_affordance(pcd_path, save_path):
    obj_name = os.path.basename(pcd_path).split('.')[0]

    # annotate point cloud (currently one label only)
    pcd = o3d.io.read_point_cloud(pcd_path)

    aff_path = os.path.join(save_path, f"{obj_name}.dummy")
    pc_annotation.annotate(np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors), aff_path)
    # extract the annotated point cloud based on the label
    label = np.loadtxt(os.path.join(str(save_path), f"{obj_name}.label"))

    label = label.astype(int)
    label = label.reshape(len(label), -1)
    for i in range(label.shape[1]):
        idx = np.nonzero(label[:, i])
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[idx])
        new_pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[idx])
        new_pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[idx])
        target_path = os.path.join(save_path, f"{obj_name}_aff{i}.pcd")
        o3d.io.write_point_cloud(target_path, new_pcd)