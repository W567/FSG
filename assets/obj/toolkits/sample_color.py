"""
Functionality: To get point cloud with color sampled from a textured mesh (.obj).

Params:            default
    num_of_points = 1024    # how many points to be sampled
    init_factor   = 5       # how many initial sampling points w.r.t. target num_of_points
    sample_color  = True    # sample color or not

Attention:
    If color is not required, directly call open3d.geometry.sample_points_poisson_disk to sample points.

Heavily based on:
http://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.sample_points_poisson_disk.html
"""

import trimesh
import open3d as o3d
import numpy as np
from math import sqrt
from queue import PriorityQueue

class Sampler:
    def __init__(self):
        # Set-up sample elimination
        self.alpha = 8
        self.beta = 0.65
        self.gamma = 1.5
        self.r_max = 0.0
        self.r_min = 0.0
        self.num_of_points = 0
        self.weights = []
        self.deleted = []
        self.mesh = None
        self.kdtree = None
        self.queue = PriorityQueue()
        self.pcd = o3d.geometry.PointCloud()
        self.new = o3d.geometry.PointCloud()

    def weight_fcn(self, d2):
        d = sqrt(d2)
        if d < self.r_min:
            d = self.r_min
        return pow(1.0 - d / self.r_max, self.alpha)

    def compute_point_weight(self, pidx0):
        [k, nbs, dists2] = self.kdtree.search_radius_vector_3d(self.pcd.points[pidx0], self.r_max)

        weight = 0.0
        for nb_index in range(len(nbs)):
            pidx1 = nbs[nb_index]
            if pidx0 == pidx1 or self.deleted[pidx1]:
                continue
            weight += self.weight_fcn(dists2[nb_index])
        self.weights[pidx0] = weight

    def init_weights(self):
        ratio = float(self.num_of_points) / len(self.pcd.points)
        self.r_max = 2 * sqrt((self.mesh.area / self.num_of_points) / (2 * sqrt(3.)))
        self.r_min = self.r_max * self.beta * (1 - pow(ratio, self.gamma))

        self.weights = [0.] * len(self.pcd.points)
        self.deleted = [False] * len(self.pcd.points)
        self.kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        
        # init weights and priority queue
        for pidx0 in range(len(self.pcd.points)):
            self.compute_point_weight(int(pidx0))
            # reverse queue into Descending Order
            self.queue.put((self.weights[pidx0] * -1.0, pidx0))

    def eliminate_sample(self):
        # sample elimination
        current_number_of_points = len(self.pcd.points)
        while current_number_of_points > self.num_of_points:
            weight, pidx = self.queue.get()
            # weight recovery
            weight *= -1.0
            # test if the entry is up to date (because of reinsert)
            if self.deleted[pidx] or weight != self.weights[pidx]:
                continue
            # delete current sample
            self.deleted[pidx] = True
            current_number_of_points -= 1
            # update weights
            [k, nbs, dists2] = self.kdtree.search_radius_vector_3d(self.pcd.points[pidx], self.r_max)
            for nb in nbs:
                self.compute_point_weight(nb)
                # reverse queue into Descending Order
                self.queue.put((self.weights[nb] * -1.0, nb))

    def update_pcd(self):
        for idx in range(len(self.pcd.points)):
            if not self.deleted[idx]:
                self.new.points.append(self.pcd.points[idx])
                self.new.colors.append(self.pcd.colors[idx])
                self.new.normals.append(self.pcd.normals[idx])

    def sample(self, path, num_of_points=1024, init_factor=5, sample_color=True):
        self.num_of_points = num_of_points
        if sample_color:
            self.mesh = trimesh.load_mesh(path)
            if self.num_of_points == 0:
                self.num_of_points = int(self.mesh.area * 5e4)
                print(f"Number of points: {self.num_of_points}")
            # https://trimsh.org/trimesh.sample.html
            cloud, face_idx, colors = trimesh.sample.sample_surface(
                                            self.mesh, init_factor * self.num_of_points, sample_color=True)
            # Reference: https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179
            bary = trimesh.triangles.points_to_barycentric(
                        triangles=self.mesh.triangles[face_idx], points=cloud)
            interp = trimesh.unitize((self.mesh.vertex_normals[self.mesh.faces[face_idx]] *
                            trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
            self.pcd.normals = o3d.utility.Vector3dVector(interp)

            # convert colors to [0, 1] from [0, 255]
            colors = np.array(colors) / 255.0
            self.pcd.points = o3d.utility.Vector3dVector(cloud)

            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # ignore transparency
            self.pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        else:
            mesh = o3d.io.read_triangle_mesh(path, True)

            if self.num_of_points == 0:
                mesh.compute_triangle_normals()
                mesh.compute_vertex_normals()

                # Get triangle vertices
                triangles = np.asarray(mesh.triangles)
                vertices = np.asarray(mesh.vertices)

                # Compute the area of each triangle
                triangle_areas = []
                for tri in triangles:
                    v0, v1, v2 = vertices[tri]
                    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                    triangle_areas.append(area)

                # Total surface area
                self.num_of_points = int(sum(triangle_areas) * 10e4)
                print(f"Number of points: {self.num_of_points}")

            pcd = mesh.sample_points_poisson_disk(self.num_of_points)
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            return pcd
        
        self.init_weights()
        self.eliminate_sample()
        self.update_pcd()
        return self.new

