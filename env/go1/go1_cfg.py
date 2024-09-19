# 有一些需要覆盖的超参就在这里进行调整
# 同时由于是具体的机器人配置文件，所以机器人的模型路径也就设置在这里

from ..base.legged_robot_config import LeggedRobotCfgPPO, LeggedRobotCfg

class Go1Cfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    
    class dof_pos_limits(LeggedRobotCfg.dof_pos_limits):
        dof_pos_limits = [
            [-0.7226, 0.7226],  # hip_joint
            [-0.7854, 3.9270],  # thigh_joint
            [-2.6075, -1.0053], # calf_joint
            [-0.7226, 0.7226],
            [-0.7854, 3.9270],
            [-2.6075, -1.0053],
            [-0.7226, 0.7226],
            [-0.7854, 3.9270],
            [-2.6075, -1.0053],
            [-0.7226, 0.7226],
            [-0.7854, 3.9270],
            [-2.6075, -1.0053]
        ]

    class asset(LeggedRobotCfg.asset):
        xml_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        body_name = []

    class control(LeggedRobotCfg.control):
        kp = 0.6
        kv = 0.08
    