# 创建训练 go1 行走的具体环境
# 需要继承上一个已经创建的 mujoco 环境

from ..base.mujoco_env import MujocoEnv

class WalkEnv(MujocoEnv):
    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)

        self.cfg = cfg