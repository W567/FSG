import os
import yaml
import rospkg
import numpy as np
from typing import List, Tuple
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

from utils.bcolors import BColors

ros_package = rospkg.RosPack()

def init_config(
        robot: str,
        pkg: str="fsg"
) -> dict:
    yaml_path = f"{ros_package.get_path(pkg)}/config/{robot}/autows.yaml"
    assert os.path.isfile(yaml_path), f"No config file found at {yaml_path}"
    config = yaml.safe_load(open(yaml_path, 'r'))
    print(f"[CONFIG] Loading config file from {yaml_path}")
    return config


def get_config_value(
        config: dict,
        key: str,
        default: any=None
) -> any:
    """
        Get the value of a key from the config file
        :param config: the config file
        :param key: the key to get the value of
            palm_name: the name of the palm
            tip_namelist: the list of tip names
            init_pose: the initial pose of the robot
            underactuated: the underactuated joints
        :param default: default value if the key is not found
        :return: the value of the key
    """
    if key in config.keys():
        value = config[key]
        print(f"[CONFIG] {key}: {value}")
    else:
        value = default
        print(f"[CONFIG] No config for {key}, using default value: {value}")
    return value


def get_config_angle_range_list(
        joint_list,
        angle_range_list,
        type_list
) -> Tuple[List[float], List[float], List[float]]:
    joint_names = [joint.name for joint in joint_list]
    joint_names = [joint_names[i] for i in range(len(joint_names)) if type_list[i] != 0]
    min_list = [angle_range_list[name]['min'] if name in angle_range_list.keys() else -100 for name in joint_names]
    max_list = [angle_range_list[name]['max'] if name in angle_range_list.keys() else  100 for name in joint_names]
    step_list = [angle_range_list[name]['step'] if name in angle_range_list.keys() else 0 for name in joint_names]
    return min_list, max_list, step_list


# https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
def indent(
        elem: ET.Element,
        level: int=0
) -> None:
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def output_ik_palm(
        path: str,
        new_link: ET.Element,
        new_joint: ET.Element
) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    for link_element in root.iter('link'):
        if link_element.attrib['name'] == 'ik_palm':
            return
    root.append(new_link)
    root.append(new_joint)
    indent(root)
    tree.write(path, xml_declaration=True, encoding='utf-8', method="xml")


def ik_palm(
        palm_rotmat: np.ndarray,
        palm_link_name: str,
        urdf_path: str,
        xacro_path: str
) -> None:
    # may cause gimbal lock, but the result with warning is still correct
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html
    r = R.from_matrix(np.linalg.inv(palm_rotmat)[:3, :3]).as_euler('xyz', degrees=False)
    link_string = '''
<link name="ik_palm" />'''
    joint_string = f'''
<joint name="ik_palm_joint" type="fixed">
<parent link="{palm_link_name}" />
<child link="ik_palm" />
<origin xyz="0 0 0" rpy="{r[0]} {r[1]} {r[2]}" />
</joint>\n'''
    new_link = ET.fromstring(link_string)
    new_joint = ET.fromstring(joint_string)

    output_ik_palm(urdf_path, new_link, new_joint)

    with open(xacro_path, 'r') as f:
        xacro_string = f.read()
    try:
        ns_content = xacro_string.split('xmlns:xacro="')[1]
        ns = ns_content.split('"')[0]
        ET.register_namespace('xacro', ns)
    except IndexError:
        print(f"{BColors.OKRED}[IK_PALM] ERROR: No namespace xmlns:xacro found in xacro file{BColors.ENDC}")
        exit()

    output_ik_palm(xacro_path, new_link, new_joint)
    print(f"{BColors.OKGREEN}\n==== Elements added to xacro & urdf files: ===={BColors.ENDC}")
    print(link_string)
    print(joint_string)


def generate_yaml(
        pc_namelist: List[str],
        ws_pc_paths: List[str],
        link_pc_paths: List[str],
        tip_intervals: List[float],
        jnt_len_max: int,
        palm_min_max: List[float],
        config_path: str
) -> None:
    finger_names = [filename.split('/')[-1] for filename in ws_pc_paths]

    ws_pc_relative_paths = []
    for pc_name in pc_namelist:
        if pc_name is None:
            ws_pc_relative_paths.append("None")
            continue
        for ws_pc_path in ws_pc_paths:
            if pc_name in ws_pc_path:
                ws_pc_relative_paths.append(ws_pc_path.split('fsg/')[1])
                break

    link_pc_relative_paths = []
    i = 0
    for pc_name in pc_namelist:
        if pc_name is None:
            link_pc_relative_paths.append("None")
            continue
        link_pc_relative_paths.append(link_pc_paths[i].split('fsg/')[1])
        i += 1

    # ws_pc_relative_paths = [filename.split('fsg/')[1] for filename in ws_pc_paths]
    print(f"{BColors.OKGREEN}[FIN_INFO] Finger name sequence is: {finger_names}{BColors.ENDC}")
    # seq = []
    # while (len(seq) != len(finger_names)):
    #     seq = input("Please input the finger priorities (highest - 0, 1, 2...), split with space\n"
    #                 ">> ")
    #     seq = [int(x) for x in seq.split(' ')]
    # sorted_pairs = sorted(zip(seq, ws_pc_relative_paths))
    # sorted_ws_pc_paths = [x[1] for x in sorted_pairs]

    print(f"{BColors.OKGREEN}[FIN_INFO] Finger tip intervals are: {tip_intervals}{BColors.ENDC}")

    fsg_config = {
        "finger_num": int(len(finger_names)),
        "finger_angle_num": int(jnt_len_max),
        "palm_min_max": palm_min_max}

    pc_path_config = {
        "ws_pc_paths": ws_pc_relative_paths,
        "link_pc_paths": link_pc_relative_paths,
    }

    with open(config_path + 'hand_info.yaml', 'w') as f:
        yaml.dump(fsg_config, f, default_flow_style=False)
    with open(config_path + 'pc_path.yaml', 'w') as f:
        yaml.dump(pc_path_config, f, default_flow_style=False)
