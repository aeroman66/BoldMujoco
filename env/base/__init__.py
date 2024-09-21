# 这里还是决定参考 isaac 的方法，将 config 和 task 分开来写。
from .mujoco_env import MujocoEnv
from .legged_robot_config import LeggedRobotCfgPPO, LeggedRobotCfg