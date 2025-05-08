import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

from utils.bcolors import BColors


def vis_pcd(pc_list, co_size, window_name):
    print(f"[VIS] {window_name}, press 'q' to continue")
    co = o3d.geometry.TriangleMesh.create_coordinate_frame(size=co_size, origin=[0, 0, 0])

    o3d_pc_list = []
    colors = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0.5, 0],
              [1, 1, 0]]
    for i, pc in enumerate(pc_list):
        pcd = o3d.geometry.PointCloud()
        if isinstance(pc, o3d.geometry.PointCloud):
            pcd.points = pc.points
        else:
            try:
                pcd.points = o3d.utility.Vector3dVector(pc)
            except:
                raise ValueError(f"Invalid point cloud data type: {type(pc)}")
        if i < len(colors):
            color = colors[i]
        else:
            color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        o3d_pc_list.append(pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, visible=True)
    vis.get_render_option().background_color = [1, 1, 1]
    vis.add_geometry(co)
    for pc in o3d_pc_list:
        vis.add_geometry(pc)
    vis.run()


def vis_result(pcd_path, pcd_namelist, ext, window_name):
    pcd_list = []
    for pcd_name in pcd_namelist:
        filename = Path(pcd_path) / f"{pcd_name}{ext}"
        pcd = o3d.io.read_point_cloud(str(filename))
        pcd_list.append(pcd)
    print(f"[VIS_RESULT] Number of points: {[len(pcd.points) for pcd in pcd_list]}")
    vis_pcd(pcd_list, 0.1, window_name)


def pc_no_hidden(pcd, center, extents, forward, upward, vis=False):
    """
    Filter out hidden points

    http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Hidden-point-removal
    """
    diameter = np.linalg.norm(extents)
    radius = diameter * 100
    camera = forward * diameter + center + upward * diameter * 0.5
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if vis:
        bounding_box = pcd.get_axis_aligned_bounding_box()
        axes_size = np.max(bounding_box.get_extent()) * 0.5
        vis_pcd([pcd,[camera]], axes_size, "candidates")

    return pcd


def add_attribute(file_path, jnt_len):
    _angle_fields = ' '.join([f"a{i}" for i in range(jnt_len)])
    _size = ' '.join(['4'] * jnt_len)
    _type = ' '.join(['F'] * jnt_len)
    _count = ' '.join(['1'] * jnt_len)

    with open(file_path, "r+") as f:
        width = len(f.readlines())
        f.seek(0,0)
        data = f.read()
        f.seek(0,0)

        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS x y z normal_x normal_y normal_z curvature xx xy xz zx zy zz {_angle_fields}\n")
        f.write(f"SIZE 4 4 4 4 4 4 4 4 4 4 4 4 4 {_size}\n")
        f.write(f"TYPE F F F F F F F F F F F F F {_type}\n")
        f.write(f"COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 {_count}\n")
        f.write(f"WIDTH {width}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {width}\n")
        f.write("DATA ascii\n")
        f.write(data)


def find_cluster_indices(pcd_tree, query_point, radius, cluster_indices):
    [_, indices, _] = pcd_tree.search_radius_vector_3d(query_point, radius)
    new_indices = [index for index in indices if index not in cluster_indices]
    return new_indices


def clustering(pcd, radius=0.005, min_num=50):
    clusters = []
    while len(pcd.points) > min_num:
        queue, cluster_indices = [0], [0]
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        while len(queue) > 0:
            query = queue.pop(0)
            new_indices = find_cluster_indices(pcd_tree, pcd.points[query], radius, cluster_indices)
            queue += new_indices
            cluster_indices += new_indices
        if len(cluster_indices) > min_num:
            clusters.append(pcd.select_by_index(cluster_indices))
        pcd = pcd.select_by_index([i for i in range(len(pcd.points)) if i not in cluster_indices])

    clusters = sorted(clusters, key=lambda x: len(x.points), reverse=True)
    return clusters

def get_max_cluster(pcd, vis):
    """
    Get the largest cluster from point cloud
    """
    clusters = clustering(pcd, radius=0.003, min_num=30)
    if vis:
        vis_pcd(clusters, 0.01, "clusters")
    return clusters[0]


def get_query_point(pcd, forward, input_query=None, vis=False):
    if input_query is None:
        tmp = np.where(np.inner(pcd.normals, forward) > 0.5)[0]
        forward_cluster = pcd.select_by_index(tmp)
        input_query = forward_cluster.get_center().reshape((3,1))
    else:
        input_query = np.array(input_query).reshape((3,1))
    query = get_neighbor(pcd, input_query, vis)
    return query


def get_average_normal_angle(neighbor_normals):
    ave_normal = np.mean(neighbor_normals, axis=0)
    ave_normal = ave_normal / np.linalg.norm(ave_normal)
    angles = np.arccos(np.clip(np.inner(neighbor_normals, ave_normal), a_min=-1, a_max=1))
    ave_angle = np.rad2deg(np.mean(angles))
    return ave_angle


def get_neighbor(pcd, query, vis):
    tree = o3d.geometry.KDTreeFlann(pcd)
    [_, idx, _] = tree.search_knn_vector_3d(query, 10)
    neighbor_pcd = pcd.select_by_index(idx)

    neighbor_points = np.asarray(neighbor_pcd.points)
    neighbor_normals = np.asarray(neighbor_pcd.normals)
    ave_angle = get_average_normal_angle(neighbor_normals)

    # TODO get convex hull of the original mesh and sampling on the convex hull is likely to be more general
    if ave_angle < 10: # TODO threshold adjustment
        diff = (neighbor_points - query.reshape((1,3)))
        diff = np.linalg.norm(diff, axis=1)
        mid_idx = np.argmin(diff)

        neighbor_pos = neighbor_points[mid_idx]
        neighbor_normal = neighbor_normals[mid_idx]
    else:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.0003,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        if vis:
            vis_pcd([inlier_cloud, outlier_cloud], 0.01, "plane segmentation")

        plane_point = np.array([1, 1, (-plane_model[3] - plane_model[0] - plane_model[1]) / plane_model[2]])
        vector_to_point = query.reshape(-1) - plane_point
        proj = vector_to_point - np.inner(vector_to_point, plane_model[:3]) * plane_model[:3]
        neighbor_pos = plane_point + proj
        neighbor_normal = plane_model[:3]
        if np.inner(neighbor_normal, np.asarray(inlier_cloud.normals)[0]) < 0:
            neighbor_normal = -neighbor_normal

    neighbor = o3d.geometry.PointCloud()
    neighbor.points = o3d.utility.Vector3dVector(np.array([neighbor_pos]))
    neighbor.normals = o3d.utility.Vector3dVector(np.array([neighbor_normal]))
    if vis:
        vis_pcd([pcd, neighbor_pcd, [query], neighbor], 0.01, "query point (orange)")
    return neighbor


# https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


def get_point_info(full_pcd, query, upward, normal_threshold, depth_threshold, shape='circle', vis=False, manual=False):
    """
    Get contact face (rectangle) centered at the query point
    """
    query_point = np.asarray(query.points)[0]
    query_normal = -np.asarray(query.normals)[0]

    trans_ps, rank2_ps, tf, x_dir, z_dir = proj_trans(full_pcd, query_point, query_normal, upward, normal_threshold, depth_threshold, vis)

    if shape == 'rect':
        x_len, z_len = max_rect_len(trans_ps, vis)
        if x_len == 0 or z_len == 0:
            return None, 0, 0, None

        corners = np.array([[x_len, z_len], [x_len, -z_len], [-x_len, z_len], [-x_len, -z_len]])

        query_left = query_point + x_len * x_dir
        query_right = query_point - x_len * x_dir
        query_up = query_point + z_len * z_dir
    elif shape == 'circle':
        radius = max_r(trans_ps, vis)

        circle_x, circle_y = generate_circle_points([0, 0], radius, 100)
        corners = np.column_stack((circle_x, circle_y))

        query_left = query_point + radius * x_dir
        query_right = query_point - radius * x_dir
        query_up = query_point + radius * z_dir

        x_len = radius
        z_len = radius
    elif shape == 'point':
        radius = max_r(trans_ps, vis)

        query_left = query_point + radius * x_dir
        query_right = query_point - radius * x_dir
        query_up = query_point + radius * z_dir

        x_len = 0
        z_len = 0
        corners = []
    else:
        raise ValueError(f"Invalid shape: {shape}")

    query_points = [query_left, query_right, query_up]

    if len(corners) > 2:
        if vis or manual:
            flag = in_hull(rank2_ps, corners)
            inlier = np.where(flag)[0]
            selected_pcd = full_pcd.select_by_index(inlier)
            remained_pcd = full_pcd.select_by_index(inlier, invert=True)
            vis_pcd([selected_pcd, remained_pcd], 0.01, "partial")
            if manual:
                while True:
                    adopt = input(f"{BColors.OKBLUE}Adopt the selected points (Red points)? [y/n]\n>> {BColors.ENDC}")
                    if adopt == 'n':
                        return None, 0, 0, None
                    elif adopt == 'y':
                        break
                    else:
                        raise ValueError("Invalid input")
    return tf, x_len, z_len, query_points


def generate_circle_points(center, radius, num_points):
    """Generate points around a circle."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    return center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)


def distance_to_edge(p1, p2, origin):
    # Calculate the distance from the origin to the line defined by p1 and p2
    numerator = abs((p2[0] - p1[0]) * (p1[1] - origin[1]) - (p1[0] - origin[0]) * (p2[1] - p1[1]))
    denominator = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return numerator / denominator


def max_inscribed_circle_radius(hull, points):
    min_distance = float('inf')
    
    for i in range(len(hull.vertices)):
        p1 = points[hull.vertices[i]]
        p2 = points[hull.vertices[(i + 1) % len(hull.vertices)]]
        distance = distance_to_edge(p1, p2, (0, 0))
        min_distance = min(min_distance, distance)
    
    return min_distance


def plot_hull(hull, pcd_xz):
    plt.plot(pcd_xz[:, 0], pcd_xz[:, 2], 'o')
    for simplex in hull.simplices:
        plt.plot(pcd_xz[simplex, 0], pcd_xz[simplex, 2], 'k-')
    plt.plot(pcd_xz[hull.vertices, 0], pcd_xz[hull.vertices, 2], 'b--', lw=2)


def max_r(pcd_xz, vis):
    rank2_ps = np.concatenate((pcd_xz[:, 0:1], pcd_xz[:, 2:3]), axis=-1)

    hull = ConvexHull(rank2_ps)
    min_distance = max_inscribed_circle_radius(hull, rank2_ps)
    # TODO failed to plot the figure continuously
    # if vis:
    #     plt.figure(num='Contact Face (Max circle)', clear=True)
    #     plot_hull(hull, pcd_xz)
    #     circle = plt.Circle((0, 0), min_distance, color='red', fill=False, linestyle='--', linewidth=1.5, label='Max inscribed circle')
    #     plt.gca().add_artist(circle)
    #     plt.scatter(0, 0, color='red', marker='o', label='Center (origin)')
    #     plt.show()

    return min_distance


def max_rect_len(pcd_xz, vis):
    """
    Get rectangle length along x and z axis with maximum rectangle area
    """
    rank2_ps = np.concatenate((pcd_xz[:, 0:1], pcd_xz[:, 2:3]), axis=-1)
    hull = ConvexHull(rank2_ps)

    vertices = np.array(rank2_ps[hull.vertices])
    vertices_1 = vertices * np.array([1, -1])
    vertices_2 = vertices * np.array([-1, 1])
    vertices_3 = vertices * np.array([-1, -1])

    flag1 = in_hull(vertices_1, vertices)
    flag2 = in_hull(vertices_2, vertices)
    flag3 = in_hull(vertices_3, vertices)
    flag = flag1 & flag2 & flag3

    areas = abs(np.multiply(np.multiply(vertices[:, 0], vertices[:, 1]), flag))
    max_index = np.argmax(areas)
    # avoid unpredictable result when areas are zeros
    if areas[max_index] < 1e-6:
        return 0, 0
    
    x_len = abs(vertices[max_index][0])
    z_len = abs(vertices[max_index][1])

    if vis:
        plt.figure(num='Contact Face (Max rectangle)')
        plot_hull(hull, pcd_xz)
        corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]]) * np.array([[x_len, z_len]])
        plt.plot(corners[:,0], corners[:,1], 'r--', lw=2)
        plt.show()

    return x_len, z_len


def proj_trans(full_pcd, query_point, query_normal, upward, normal_threshold, depth_threshold, vis):
    """
    Project partial point cloud to the plane defined by query point and query normal
    Transform the projected point cloud to X-O-Z plane with query point at the origin
    """
    complete_points = np.asarray(full_pcd.points)
    complete_normals = np.asarray(full_pcd.normals)

    x_dir = np.cross(query_normal, upward)
    x_dir = x_dir / np.linalg.norm(x_dir)
    z_dir = np.cross(x_dir, query_normal)
    
    delta = (complete_points - query_point).dot(query_normal)
    inner_normal = complete_normals.dot(-query_normal)
    inlier = np.where((inner_normal > normal_threshold) & (delta < depth_threshold))
    points = complete_points[inlier]

    tf = np.identity(4)
    tf[:3, :3] = np.array([x_dir, query_normal, z_dir]).transpose()
    delta = np.mean(delta[inlier])
    tf[:3, 3] = query_point + delta * query_normal

    proj_matrix = np.eye(3) - np.outer(query_normal, query_normal)
    proj_points = (points - query_point).dot(proj_matrix.T) + query_point

    rot_matrix = np.linalg.inv(np.array([x_dir, query_normal, z_dir]).transpose())
    trans_ps = np.dot(rot_matrix, (proj_points - query_point).T).T

    proj_points_partial = (complete_points - query_point).dot(proj_matrix.T) + query_point
    trans_ps_partial = np.dot(rot_matrix, (proj_points_partial - query_point).T).T
    rank2_ps = np.concatenate((trans_ps_partial[:,0:1], trans_ps_partial[:,2:3]), axis=-1)

    if vis:
        vis_pcd([complete_points, proj_points], 0.01, "projection")
        vis_pcd([complete_points, trans_ps], 0.01, "transform")
        vis_pcd([complete_points, points, [query_point]], 0.01, "selection")
    
    return trans_ps, rank2_ps, tf, x_dir, z_dir

