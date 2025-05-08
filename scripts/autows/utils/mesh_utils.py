import trimesh
import numpy as np
from utils.bcolors import BColors


def get_palm_bounds(
        mesh_path: str,
        mesh_scale: np.ndarray,
        palm_rotmat: np.ndarray
) -> (list, np.ndarray, np.ndarray):
    palm_center, palm_extents, palm_bounds = get_mesh_size(mesh_path, mesh_scale)
    min_bound = palm_bounds[0]
    max_bound = palm_bounds[1]
    palm_min_bound = palm_rotmat[:3, :3] @ min_bound
    palm_max_bound = palm_rotmat[:3, :3] @ max_bound
    for i in range(3):
        if palm_min_bound[i] > palm_max_bound[i]:
            palm_min_bound[i], palm_max_bound[i] = palm_max_bound[i], palm_min_bound[i]
    palm_min_bound = [round(float(x), 4) for x in palm_min_bound]
    palm_max_bound = [round(float(x), 4) for x in palm_max_bound]
    palm_min_max = [palm_min_bound[0], palm_max_bound[0],
                    palm_min_bound[1], 1.0,
                    palm_min_bound[2], palm_max_bound[2]]
    return palm_min_max, palm_center, palm_extents


def input_direction(direction, manual=False):
    direction_vectors = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
        '-x': np.array([-1, 0, 0]),
        '-y': np.array([0, -1, 0]),
        '-z': np.array([0, 0, -1])
    }
    vector = None
    while True:
        if manual:
            local_direction = input(f"{BColors.OKBLUE}Which direction as {direction} under tip frame? [x/y/z/-x/-y/-z/customize(c)]\n"
                                    f">> {BColors.ENDC}")
        else:
            if 'forward' in direction:
                local_direction = '-y'
            elif 'upward' in direction:
                local_direction = 'z'
            else:
                local_direction = None

        if local_direction in direction_vectors.keys():
            vector = direction_vectors[local_direction]
        elif local_direction == 'c':
            while True:
                vec = np.array([float(x) for x in input(f"{BColors.OKBLUE}Customized {direction} vector, 3d required, split with space\n"
                                                        f">> {BColors.ENDC}").split(' ')])
                if vec.size != 3:
                    print(f"{BColors.OKRED}Invalid input, vector3d required.{BColors.ENDC}")
                    continue
                vector = vec / np.linalg.norm(vec)
                break
        else:
            print(f"{BColors.OKRED}Invalid input, choose from [x/y/z/-x/-y/-z/customize(c)].{BColors.ENDC}")
            continue
        break
    return vector


def get_mesh_size(mesh_path, mesh_scale):
    mesh = trimesh.load(mesh_path, force='mesh')
    matrix = np.diag(mesh_scale)
    mesh.apply_transform(matrix)
    return mesh.bounding_box.centroid, mesh.extents, mesh.bounds


def get_tip_interval(upward, forward, mesh_extents):
    sidewards = np.cross(upward, forward)
    tip_interval = round(np.fabs(np.inner(mesh_extents, sidewards)), 5)
    return tip_interval
