import sympy
import itertools
import numpy as np
from typing import List, Tuple
from copy import deepcopy


def check_duplication(name: str, namelist: list):
    """
    Check if name exists in namelist
    """
    try:
        idx = namelist.index(name)
    except ValueError:
        idx = None
    return idx


def remove_similar_rows(arr, threshold=0.001):
    unique_rows = [arr[0]]  # Start with the first row
    for i in range(1, len(arr)):
        # Compute the Euclidean distance between the current row and previous row
        distance = np.linalg.norm(arr[i] - arr[i-1])
        # If the distance to all previous unique rows is greater than the threshold, keep this row
        if distance >= threshold:
            unique_rows.append(arr[i])
    return np.array(unique_rows)


def fk_calc(joint_list, tf_list, joint_type_list, mimic_multiplier_list, mimic_offset_list, var_list
) -> Tuple[any, List[str], List[any]]:
    """
    Forward kinematics calculation
    From palm link to tip link origin
    """
    cal_var_list = []
    res = sympy.eye(4)
    link_func_list = []
    for j, tf in enumerate(tf_list):
        var = var_list[j]
        mat = sympy.eye(4)
        if var not in cal_var_list and var != 'fixed':
            cal_var_list.append(var)

        if joint_type_list[j] == 1: # revolute
            mat[:3, :3] = sympy_rotmat(joint_list[j].axis,
                                        mimic_multiplier_list[j]*var+mimic_offset_list[j])
        elif joint_type_list[j] == 2: # prismatic
            mat[:3, 3] = (mimic_multiplier_list[j]*var+mimic_offset_list[j]) * joint_list[j].axis
        elif joint_type_list[j] == 0: # fixed
            mat = sympy.eye(4)
        else:
            raise NotImplementedError("Only revolute, prismatic and fixed joints are implemented")
        res = res @ tf @ mat
        link_func_list.append([deepcopy(cal_var_list), sympy.lambdify(cal_var_list, res)])
    res_func = sympy.lambdify(cal_var_list, res)
    return res_func, cal_var_list, link_func_list


def get_variables(joint_list, joint_type_list, mimic_joint_list):
    # set variables for each active & revolute joint
    # if joint is mimicking another joint, use the same variable
    var_list, min_list, max_list = [], [], []
    joint_names = [joint.name for joint in joint_list]
    for j, joint in enumerate(joint_list):
        if joint_type_list[j] == 1: # revolute
            if mimic_joint_list[j] == 'None': # active joint
                var = sympy.symbols('theta' + str(j))
                min_list.append(np.rad2deg(joint.limit.lower))
                max_list.append(np.rad2deg(joint.limit.upper))
            else:
                mimic_joint = mimic_joint_list[j]
                idx = joint_names.index(mimic_joint)
                var = sympy.symbols('theta' + str(idx))
        elif joint_type_list[j] == 2: # prismatic
            if mimic_joint_list[j] == 'None': # active joint
                var = sympy.symbols('d' + str(j))
                min_list.append(joint.limit.lower * 1000.0) # mm
                max_list.append(joint.limit.upper * 1000.0)
            else:
                mimic_joint = mimic_joint_list[j]
                idx = joint_names.index(mimic_joint)
                var = sympy.symbols('d' + str(idx))
        elif joint_type_list[j] == 0: # fixed
            var = 'fixed'
        else:
            raise NotImplementedError("Only revolute, prismatic and fixed joints are implemented")
        var_list.append(var)
    return var_list, min_list, max_list


def sympy_rotmat(axis: List[float], theta: float) -> np.ndarray:
    q = sympy.Quaternion.from_axis_angle(axis, theta)
    return np.array(q.to_rotation_matrix())


def generate_combination(min_list, max_list, step_list):
    all_combinations = []
    # Iterate over each minimum-maximum pair
    for min_val, max_val, step in zip(min_list, max_list, step_list):
        # Generate values between min and max with the specified step
        values = [i for i in range(int(min_val), int(max_val + 1), step)]
        all_combinations.append(values)

    # Use itertools.product to generate all possible combinations
    result = list(itertools.product(*all_combinations))
    return result


# TODO only for srh condition (last joint following last second joint)
# Other different underactuated conditions are not implemented
def get_joint_angle_combination(min_list, max_list, step_list, underactuated):
    combs = []
    if underactuated == 0:
        combs = generate_combination(min_list, max_list, step_list)
    elif underactuated == 1:
        combs = generate_combination(min_list[:-1], max_list[:-1], step_list)
        combs = [comb + (comb[-1],) for comb in combs]
    return combs
