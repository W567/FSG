#/usr/bin/env python3
# Copyright 2024 JSK Laboratory Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Functionality: To get point cloud sampled from a textured mesh (.dae).

Params:            default
    num_of_points = 1024    # how many points to be sampled
    init_factor   = 5       # how many initial sampling points w.r.t. target num_of_points

Heavily based on:
http://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.sample_points_poisson_disk.html
"""

import trimesh
from trimesh.sample import sample_surface
from trimesh.triangles import points_to_barycentric
import open3d as o3d
import numpy as np
from math import sqrt
from queue import PriorityQueue

from utils.bcolors import BColors

class Sampler:
    def __init__(self):
        self.new = None
        self.pcd = None
        self.queue = None
        self.kdtree = None
        self.mesh = None
        self.deleted = None
        self.weights = None
        self.num_of_points = None
        self.r_min = None
        self.r_max = None
        self.gamma = None
        self.beta = None
        self.alpha = None

    def init_variables(self):
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
        [_, nbs, dists2] = self.kdtree.search_radius_vector_3d(self.pcd.points[pidx0], self.r_max)

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
            [_, nbs, _] = self.kdtree.search_radius_vector_3d(self.pcd.points[pidx], self.r_max)
            for nb in nbs:
                self.compute_point_weight(nb)
                # reverse queue into Descending Order
                self.queue.put((self.weights[nb] * -1.0, nb))

    def update_pcd(self):
        for idx in range(len(self.pcd.points)):
            if not self.deleted[idx]:
                self.new.points.append(self.pcd.points[idx])
                self.new.normals.append(self.pcd.normals[idx])

    def sample(self, path, scale=None, density=10e5, init_factor=5, num_points = None):
        self.init_variables()

        self.mesh = trimesh.load(path, force='mesh')
        if scale is None:
            scale = [1.0, 1.0, 1.0, 1.0]
        assert len(scale) == 4, "[Sampler] scale must be a list of 4 floats"
        matrix = np.diag(scale)
        self.mesh.apply_transform(matrix)
        if num_points is not None:
            self.num_of_points = num_points
        else:
            self.num_of_points = int(self.mesh.area * density)
        print(f"{BColors.OKYELLOW}[Sampler] {self.num_of_points} points to be sampled{BColors.ENDC}")

        # https://trimsh.org/trimesh.sample.html
        cloud, face_idx = sample_surface(self.mesh, init_factor * self.num_of_points)
        # Reference: https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179
        bary = points_to_barycentric(triangles=self.mesh.triangles[face_idx], points=cloud)
        interp = trimesh.unitize((self.mesh.vertex_normals[self.mesh.faces[face_idx]] *
                        trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
        self.pcd.normals = o3d.utility.Vector3dVector(interp)
        self.pcd.points = o3d.utility.Vector3dVector(cloud)
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        self.init_weights()
        self.eliminate_sample()
        self.update_pcd()
        return self.new