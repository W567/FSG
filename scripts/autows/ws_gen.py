import os
import subprocess
import sys

import rospkg

ros_package = rospkg.RosPack()

obj_path = ros_package.get_path('fsg') + '/assets/obj'

obj_name_list = [name.split('.pcd')[0] for name in os.listdir(obj_path) if
                      'pcd' in name and 'aff' not in name]
obj_name_list = sorted(obj_name_list)
obj_max_num = len(obj_name_list)

parent_folder = ros_package.get_path('fsg') + '/workspace'
exist_ws_pc_list = [
    folder for folder in os.listdir(parent_folder)
    if os.path.isdir(os.path.join(parent_folder, folder)) and folder.startswith('ycb')
]

for obj_name in obj_name_list:
    print(obj_name)
    if obj_name in exist_ws_pc_list:
        print(f"Skipping {obj_name}")
        continue
    keyframe_script_path = ros_package.get_path('fsg') + '/scripts/read_keyframe.py'
    keyframe_command = [
        "python",
        keyframe_script_path,
        "--robot", "srh",
        "--obj_folder", "obj",
        "--obj", obj_name,
    ]
    keyframe_result = subprocess.run(keyframe_command, text=True, stdout=sys.stdout, stderr=sys.stderr)

    ws_script_path = ros_package.get_path('fsg') + '/scripts/autows/autows.py'
    ws_command = [
        "python",
        ws_script_path,
        "--robot", "srh",
        "--folder", obj_name,
        "--separate",
        "--aug",
        "--mj_obj_body_name", obj_name,
        "--key", "1",
        "--shape", "circle"
    ]
    ws_result = subprocess.run(ws_command, text=True, stdout=sys.stdout, stderr=sys.stderr)
