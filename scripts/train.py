import os
import sys
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

from env.go1 import WalkEnv
from env.go1 import Go1Cfg
from env.reward import Reward
from env.base import LeggedRobotCfgPPO
from utils import *
from runner import OnPolicyRunner

train_cfg = LeggedRobotCfgPPO()
cfg = Go1Cfg()
reward = Reward
env = WalkEnv(cfg, reward)
log_dir = parent_dir + '/log'
# 初始化一个runner
runner = OnPolicyRunner(
    env=env,
    train_cfg=train_cfg,
    log_dir=log_dir,
)
runner.learn(num_learning_iterations=500)