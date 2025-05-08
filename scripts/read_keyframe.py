#!/usr/bin/env python
import os
import re
import argparse
import numpy as np
import lxml.etree as ET

import rosparam
import rospkg
import rospy
from pygments.lexer import default

ros_package = rospkg.RosPack()

import mujoco as mj
from mujoco import viewer
from collections import defaultdict

def parser():
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument('-r', '--robot', type=str, default='srh', help='robot name')
    local_parser.add_argument('--obj_folder', type=str, default='obj', help='object folder name')
    local_parser.add_argument('--obj', type=str, default='', help='which object to view')
    local_parser.add_argument('--vis', action='store_true', help='visualize the keyframe')
    return local_parser


class Reader:
    def __init__(self):
        self.keyframe_id = None
        self.args = parser().parse_args()

        self.viewer = None

        self.obj_path = ros_package.get_path('fsg') + '/assets/' + self.args.obj_folder
        self.keyframe_file_path = ros_package.get_path('fsg') + '/assets/obj/keyframes/' + self.args.robot + '.txt'
        self.scene_path = ros_package.get_path('fsg') + '/assets/robot/' + self.args.robot + '/xml/scene.xml'
        
        print(self.scene_path)
        self.scene_tree = ET.parse(self.scene_path)
        self.scene_root = self.scene_tree.getroot()

        self.keyframe_name = 'grasp_demo'
        self.init_keyframe()

    def init_keyframe(self):
        for i, frame in enumerate(self.scene_root.iter('key')):
            if frame.get('name') == self.keyframe_name:
                self.keyframe_id = i
                break
        print("Keyframe: %s -- %d" % (self.keyframe_name, self.keyframe_id))

    def update_keyframe(self, qpos, ctrl):
        for key in self.scene_root.iter('key'):
            if key.get('name') == self.keyframe_name:
                key.set('qpos', qpos)
                key.set('ctrl', ctrl)
                break
        self.scene_tree.write(self.scene_path)

    def switch_obj(self, obj_name):
        for ind in self.scene_root.iter('include'):
            if 'obj/' in ind.get('file'):
                path = f"../../../{self.args.obj_folder}/{obj_name}.xml"
                ind.set('file', path)
                break

    def change_obj_pose(self, obj, obj_pos, obj_quat):
        obj_xml_path = f"{self.obj_path}/{obj}.xml"
        obj_tree = ET.parse(obj_xml_path)
        obj_root = obj_tree.getroot()

        for body in obj_root.iter('body'):
            body.set('pos', obj_pos)
            body.set('quat', obj_quat)
            obj_tree.write(obj_xml_path)
            with open(obj_xml_path, 'r') as f:
                os.fsync(f.fileno())
            break

    def view(self):
        model = mj.MjModel.from_xml_path(self.scene_path)
        data = mj.MjData(model)
        mj.mj_resetDataKeyframe(model, data, self.keyframe_id)

        if self.viewer is None:
            self.viewer = mj.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=False)
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        else:
            self.viewer._sim().load(model, data, "")

        mj.mj_step(model, data)
        self.viewer.sync()


    def parse_keyframes(self):
        data = defaultdict(list)
        current_object = None
        with open(self.keyframe_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Detect object name (e.g., ycb_062_dice, ycb_065-h_cups)
                if re.match(r'^[a-zA-Z0-9_\-]+$', line):
                    current_object = line
                # Collect other data (position, orientation, or <key> XML)
                elif current_object and line != '':
                    data[current_object].append(line)

        result = defaultdict(dict)
        for obj, items in data.items():
            if self.args.obj != '' and self.args.obj != obj:
                continue
            position_strings = []
            orientation_strings = []
            qpos_strings = []
            ctrl_strings = []
            for index in range(0, len(items), 5):
                position_string = items[index]
                orientation_string = items[index + 1]
                qpos = items[index + 2]
                ctrl = items[index + 3]

                position_strings.append(position_string)
                orientation_strings.append(orientation_string)
                qpos_strings.append(qpos)
                ctrl_strings.append(ctrl)

            result[obj] = {
                'position': position_strings,
                'orientation': orientation_strings,
                'qpos': qpos_strings,
                'ctrl': ctrl_strings
            }
        return dict(sorted(result.items()))


    def read(self):
        keyframes_data = self.parse_keyframes()

        # Print the result
        for obj, content in keyframes_data.items():
            self.switch_obj(obj)
            for i in range(len(content['position'])):
                self.change_obj_pose(obj, content['position'][i], content['orientation'][i])
                qpos = content['qpos'][i]
                ctrl = content['ctrl'][i]
                self.update_keyframe(qpos , ctrl)
                if self.args.vis:
                    self.view()
                    input("Press Enter to continue...")
                else:
                    print(f"Object: {obj}")
                    return



reader = Reader()
reader.read()


