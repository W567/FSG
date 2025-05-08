#!/usr/bin/env python
import os
import time
import h5py
import rospkg
import trimesh
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.model.primitives import Axis, MeshLink, PointCloudLink

from utils import read_pose, set_robot_state

ros_package = rospkg.RosPack()

def parser():
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument('-r', '--robot', type=str, default='srh', help='robot name')
    local_parser.add_argument('--obj_folder', type=str, default='obj', help='object folder name')
    local_parser.add_argument('--file', type=str, required=True, help='hdf5 filename to be read')
    local_parser.add_argument('--obj', type=str, default='', help='which object to view')
    local_parser.add_argument('--pose_id', type=int, default=-1, help='which pose to view')
    local_parser.add_argument('--vis_all', action='store_true', help='visualize all poses in a view')
    local_parser.add_argument('--multi', action='store_true', help='visualize multiple robots')
    local_parser.add_argument('--viewer', type=str, default='pyrender', help='viewer type [pyrender, trimesh]')
    local_parser.add_argument('--record', action='store_true', help='record video and save image')
    local_parser.add_argument('--duration', type=float, default=60.0, help='duration of the video')
    local_parser.add_argument('--angle', type=float, default=360.0, help='rotation degrees in the duration')
    return local_parser

args = parser().parse_args()
if args.viewer == 'trimesh':
    from skrobot.viewers import TrimeshSceneViewer as Viewer
elif args.viewer == 'pyrender':
    from skrobot.viewers import PyrenderViewer as Viewer
else:
    raise ValueError(f"Viewer type {args.viewer} is not supported")



class ReaderGrasp:

    def __init__(self):
        self.args = parser().parse_args()

        robot = self.args.robot
        if self.args.robot == 'srh':
            self.robot_joint_names = ["rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
                                      "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
                                      "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
                                      "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
                                      "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"]
        elif self.args.robot == 'slh':
            self.robot_joint_names = ["lh_FFJ4", "lh_FFJ3", "lh_FFJ2", "lh_FFJ1",
                                      "lh_MFJ4", "lh_MFJ3", "lh_MFJ2", "lh_MFJ1",
                                      "lh_RFJ4", "lh_RFJ3", "lh_RFJ2", "lh_RFJ1",
                                      "lh_LFJ5", "lh_LFJ4", "lh_LFJ3", "lh_LFJ2", "lh_LFJ1",
                                      "lh_THJ5", "lh_THJ4", "lh_THJ3", "lh_THJ2", "lh_THJ1"]
        elif self.args.robot == 'allegro':
            self.robot_joint_names = ["FFJ_0", "FFJ_1", "FFJ_2", "FFJ_3",
                                      "MFJ_0", "MFJ_1", "MFJ_2", "MFJ_3",
                                      "RFJ_0", "RFJ_1", "RFJ_2", "RFJ_3",
                                      "THJ_0", "THJ_1", "THJ_2", "THJ_3"]
        elif self.args.robot == 'barrett':
            self.robot_joint_names = ["hand/finger_1/prox_joint", "hand/finger_1/med_joint", "hand/finger_1/dist_joint",
                                      "hand/finger_2/prox_joint", "hand/finger_2/med_joint", "hand/finger_2/dist_joint",
                                      "hand/finger_3/med_joint", "hand/finger_3/dist_joint"]
        elif self.args.robot == 'robotiq85':
            self.robot_joint_names = ["finger_joint"]
        else:
            raise ValueError(f"Robot {self.args.robot} is not supported: \n"
                              "    robot_joint_names in reader_grasp.py is not defined")

        self.folder_path = ros_package.get_path('fsg') + f"/hdf5/{self.args.robot}/"
        self.file_address = self.folder_path + self.args.file
        self.obj_path = ros_package.get_path('fsg') + f"/assets/{self.args.obj_folder}/"

        if self.args.multi:
            robot_path = ros_package.get_path('fsg') + f"/assets/robot/{robot}/urdf/{robot}.urdf"
            self.robot = RobotModelFromURDF(urdf_file=robot_path)
            robot_path1 = ros_package.get_path('fsg') + f"/assets/robot/{robot}/urdf/{robot}.urdf"
            self.robot1 = RobotModelFromURDF(urdf_file=robot_path1)
            robot_path2 = ros_package.get_path('fsg') + f"/assets/robot/{robot}/urdf/{robot}.urdf"
            self.robot2 = RobotModelFromURDF(urdf_file=robot_path2)
        else:
            robot_path = ros_package.get_path('fsg') + f"/assets/robot/{robot}/urdf/{robot}.urdf"
            self.robot = RobotModelFromURDF(urdf_file=robot_path)

        if not self.args.record and not self.args.vis_all:
            self.viewer = Viewer(resolution=(500, 500))

            self.viewer.add(self.robot)
            if self.args.multi:
                self.viewer.add(self.robot1)
                self.viewer.add(self.robot2)

            axis = Axis(axis_radius=0.001, axis_length=0.1)
            self.viewer.add(axis)
            self.viewer.show()


    def get_obj_instance(self, name):
        obj_instance_path = f"{self.obj_path}{name}.obj"
        if not os.path.exists(obj_instance_path):
            obj_instance_path = f"{self.obj_path}{name}.pcd"
            if not os.path.exists(obj_instance_path):
                print(f'Object not found: {obj_instance_path}')
                return None
            else:
                obj_pcd = o3d.io.read_point_cloud(obj_instance_path)
                points = np.asarray(obj_pcd.points)
                colors = np.asarray(obj_pcd.colors)
                point_cloud = trimesh.PointCloud(points, colors=colors)
                obj_instance = PointCloudLink(point_cloud_like=point_cloud)
        else:
            obj_mesh = trimesh.load(obj_instance_path)
            obj_instance = MeshLink(visual_mesh=obj_mesh)
        return obj_instance


    def vis_all(self):
        with h5py.File(self.file_address, 'r') as f:
            # get names of all objects
            obj_names = [name for name in f.keys()]
            print('Available object names in dataset: ', obj_names)
            for i, name in enumerate(obj_names):
                if self.args.obj != '' and self.args.obj != name:
                    continue
                print(f"Object {i}: {name}")

                obj_instance_path = f"{self.obj_path}{name}.obj"
                obj_mesh = o3d.io.read_triangle_mesh(obj_instance_path, True)
                axis_length = 1.5 * np.max(obj_mesh.get_max_bound())
                if name == 'ycb_073-g_lego_duplo':
                    axis_length /= 2.5
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=[0, 0, 0])

                group_read = f[name]
                palm_poses = group_read['palm_pose']

                # Convert to Open3D PointCloud
                palm_pose_point_cloud = o3d.geometry.PointCloud()
                palm_pose_point_cloud.points = o3d.utility.Vector3dVector(palm_poses[:, :3])
                palm_pose_point_cloud.colors = o3d.utility.Vector3dVector(
                    np.tile([0, 1, 0], (len(palm_poses), 1))  # Green color for points
                )

                points = palm_poses[:, :3]
                normals = points / np.linalg.norm(points, axis=1, keepdims=True)
                palm_pose_point_cloud.normals = o3d.utility.Vector3dVector(normals)

                if self.args.record:
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name=f"Object {name}", width=600, height=600)
                    vis.add_geometry(obj_mesh)
                    vis.add_geometry(palm_pose_point_cloud)
                    vis.add_geometry(axis)

                    # Adjust the initial view
                    view_control = vis.get_view_control()

                    front_vector = np.array([1, 1, 1.5])
                    front_vector = front_vector / np.linalg.norm(front_vector)  # Normalize the vector

                    # Set the camera to look at the origin
                    view_control.set_lookat([0, 0, 0])  # Looking at the origin
                    view_control.set_up([0, 0, 1])  # Z-up orientation
                    view_control.set_front(front_vector)  # Front of the camera pointing towards origin
                    view_control.set_zoom(0.75)  # Optional: adjust zoom level if necessary

                    vis.poll_events()
                    vis.update_renderer()

                    # Define the output video path
                    video_path = f"output_videos/{name}_rotation.mp4"
                    os.makedirs(os.path.dirname(video_path), exist_ok=True)

                    # Set the total number of frames for a 360-degree rotation at 90 degrees per second
                    total_rotation_time = 4  # 360 degrees / 90 degrees per second
                    frame_rate = 30  # Frames per second in video
                    total_frames = total_rotation_time * frame_rate

                    # Record the video
                    for frame_idx in range(total_frames):
                        # Rotate the camera around the Z-axis
                        rotation_angle = 360 * (frame_idx / total_frames)
                        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.radians(rotation_angle)])

                        # Set the camera's front direction based on the rotation
                        rotated_front = np.dot(rotation_matrix, front_vector)
                        view_control.set_front(rotated_front)

                        # Capture the screen image for the video
                        vis.poll_events()
                        vis.update_renderer()
                        vis.capture_screen_image(f"temp_frame_{frame_idx:04d}.png")

                    # Now save the images as a video using an external tool (e.g., ffmpeg)
                    os.system(f"ffmpeg -framerate {frame_rate} -i temp_frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {video_path}")
                    print(f"Video saved at: {video_path}")

                    # Cleanup temporary frames
                    for frame_idx in range(total_frames):
                        os.remove(f"temp_frame_{frame_idx:04d}.png")

                    # Save the image
                    save_path = f"output_images/{name}_all_visualization.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    vis.capture_screen_image(save_path)
                    print(f"Image saved at: {save_path}")
                    vis.destroy_window()
                    del view_control
                else:
                    o3d.visualization.draw_geometries([obj_mesh, palm_pose_point_cloud, axis])
                    input("Press Enter to continue...")


    def vis(self):
        with h5py.File(self.file_address, 'r') as f:
            # get names of all objects
            obj_names = [name for name in f.keys()]
            print('Available object names in dataset: ', obj_names)
            for i, name in enumerate(obj_names):
                if self.args.obj != '' and self.args.obj != name:
                    continue
                print(f"Object {i}: {name}")

                obj_instance = self.get_obj_instance(name)
                if obj_instance is None:
                    continue
                if self.args.record:
                    if self.args.viewer != 'pyrender':
                        print(f"Use pyrender viewer for recording")
                        exit(0)
                    pose_ids = np.arange(1000)
                    self.viewer = Viewer(resolution=(800, 800), render_flags={"run_in_thread": True, "record": True})
                    self.viewer.add(self.robot)
                    self.viewer.add(obj_instance)
                    self.viewer.show()
                else:
                    self.viewer.add(obj_instance)

                group_read = f[name]
                if self.args.multi:
                    input_tmp = 'r'
                    while input_tmp == 'r':
                        # get three random id
                        ids = np.random.choice(group_read['hand_pose'].shape[0], 3, replace=False)
                        hand_pose, palm_pose = read_pose(group_read, ids[0])
                        set_robot_state(self.robot, self.robot_joint_names, hand_pose, palm_pose, self.args.robot)
                        hand_pose, palm_pose = read_pose(group_read, ids[1])
                        set_robot_state(self.robot1, self.robot_joint_names, hand_pose, palm_pose, self.args.robot)
                        hand_pose, palm_pose = read_pose(group_read, ids[2])
                        set_robot_state(self.robot2, self.robot_joint_names, hand_pose, palm_pose, self.args.robot)
                        self.viewer.redraw()
                        input_tmp = input(f"---- q to exit, r to redraw, others to continue: ")
                        if input_tmp == 'q':
                            return
                        elif input_tmp == 'r':
                            continue
                        else:
                            self.viewer.set_camera(angles=[np.pi/4.0, 0, 3 * np.pi/4.0],
                                                   distance=0.4,
                                                   center=[0,0,0],
                                                   resolution=None,
                                                   fov=None)
                            if self.args.viewer == 'trimesh':
                                save_image = input(f"---- save image? (y to save) :")
                                if save_image == 'y':
                                    os.makedirs("output_images", exist_ok=True)
                                    image_path = f"output_images/{name}_multi_visualization.png"
                                    self.viewer.save_image(image_path)
                            input(f"---- next? ")
                else:
                    if self.args.record:
                        hand_pose, palm_pose = read_pose(group_read, pose_ids[0])
                        set_robot_state(self.robot, self.robot_joint_names, hand_pose, palm_pose, self.args.robot)

                        gif_path = f"output_gifs/{name}.gif"
                        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                        self.viewer.redraw()
                        self.viewer.set_camera(angles=[np.pi/4.0, 0, 3 * np.pi/4.0],
                                               distance=0.4,
                                               center=[0,0,0],
                                               resolution=None,
                                               fov=None)
                        total_rotation_time = self.args.duration  # seconds
                        frame_rate = 30  # Frames per second in video
                        total_frames = int(total_rotation_time * frame_rate)

                        for frame_idx in range(total_frames):
                            rotation_angle = np.deg2rad(self.args.angle * (frame_idx / total_frames))
                            self.viewer.set_camera(angles=[np.pi / 4.0, 0, 3 * np.pi / 4.0 + rotation_angle],
                                                   distance=0.4,
                                                   center=[0, 0, 0],
                                                   resolution=None,
                                                   fov=None)
                            time.sleep(1 / frame_rate)
                            if frame_idx % 30 == 0:
                                hand_pose, palm_pose = read_pose(group_read, pose_ids[(int(frame_idx / 15)) % group_read['hand_pose'].shape[0]])
                                set_robot_state(self.robot, self.robot_joint_names, hand_pose, palm_pose, self.args.robot)
                                self.viewer.redraw()
                        self.viewer.close_external()
                        self.viewer.save_gif(gif_path)
                    else:
                        for idx in range(group_read['hand_pose'].shape[0]):
                            hand_pose, palm_pose = read_pose(group_read, idx)
                            set_robot_state(self.robot, self.robot_joint_names, hand_pose, palm_pose, self.args.robot)
                            self.viewer.redraw()

                            if input(f"---- {idx} next? (q to exit): ") == 'q':
                                return

                self.viewer.delete(obj_instance)


    def exe(self):
        if self.args.vis_all:
            self.vis_all()
        else:
            self.vis()

if __name__ == "__main__":
    reader = ReaderGrasp()
    reader.exe()
