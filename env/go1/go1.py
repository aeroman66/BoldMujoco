# 创建训练 go1 行走的具体环境
# 需要继承上一个已经创建的 mujoco 环境
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..base.mujoco_env import MujocoEnv
from utils import *

from go1_cfg import Go1Cfg
from ..reward.reward import Reward

class WalkEnv(MujocoEnv):
    def __init__(self, cfg=Go1Cfg, reward=Reward) -> None:
        super().__init__(cfg)

        self.cfg = cfg
        self.reward = reward

        self._parse_cfg()
        self._prepare_reward_function()
        self._init_tensor()
        self._init_act_obs()

# *************************************Initialization*********************************************
    def _parse_cfg(self):
        """将参数类中的参数转化为字典
        """
        self.reward_scales = class_to_dict(self.cfg.reward.scales)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # 先把 0. 去掉，避免计算多余的项
        for key, scale in list(self.reward_scales.items()):
            if scale == 0.:
                self.reward_scales.pop(key) # 这个键对应的值会被返回
            else:
                self.reward_scales[key] *= self.dt
        # 将 reward 函数和 reward 名字分别存入两个列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == 'termination':
                continue
            self.reward_names.append(name)
            name = '_reward_' + name # 感觉这么操作很危险，名字写错就完蛋了，能不能加点纠错机制
            self.reward_functions.append(getattr(self, name)) # self.reward 的属性应该也在 __dict__ 中，所以可以用 getattr 获取

        # reward episode sums
        self.episode_sums = {name: 0. for name in self.reward_scales.keys()} # 字典推导
        print('episode_sums : ',self.episode_sums)
        print('reward_functions : ', self.reward_functions)
        print('reward_names : ', self.reward_names)

    def _init_tensor(self):
        """对训练中会用到的各个张量进行定义和初始化
        """
        self.step_counter = 0.
        self.noise_scale_vec = self._get_noise_scale_vec()
        self._pos_his = np.zeros([self.cfg.env.num_actions, 3])
        self._vel_his = np.zeros([self.cfg.env.num_actions, 3])
        self.dof_pos_limits = np.array(self.cfg.dof_pos_limits.dof_pos_limits)
        self._keep_time = 0

        # 获取 data 中的一些数据
        self._qpos = np.array(self.data.qpos)
        self._qvel = np.array(self.data.qvel)
        self._joint_pos = self._qpos.copy()[-self.cfg.env.num_actions:]
        self._joint_vel = self._qvel.copy()[-self.cfg.env.num_actions:]
        self._joint_torque = np.array(self.data.ctrl[:self.cfg.env.num_actions])
        self._base_quat = np.array([self._qpos[4], self._qpos[5], self._qpos[6], self._qpos[3]])
        self._base_rotate = R.from_quat(self._base_quat)
        self._rotation_mat_b2w = self._base_rotate.as_matrix()
        self._rotation_mat_w2b = self._rotation_mat_b2w.T.copy()
        self._base_pos = self._qpos.copy()[0:3]

        self._base_lin_vel_b = (self._rotation_mat_w2b @ self._qvel[0:3])
        self._base_ang_vel_b = self._qvel.copy()[3:6]
        self._base_ang_vel_w = (self._rotation_mat_w2b.T @ self._qvel[3:6])
        self._last_con_list = [False, False, False, False]
        self.feet_air_time = np.array([0., 0., 0., 0.])
        self.feet_con_time = np.array([0., 0., 0., 0.])
        self._last_action = np.zeros(self.n)
        self._action = np.zeros(self.n)
        self.command = np.array([0.5, 0.0, 0]) # command 应该找一个更好的地方安置

        self._observe_targets = np.zeros([self.cfg.env.num_history_obs, 3])
        # ====== obs_scale ======
        self.lin_vel_scale = self.cfg.normalization.obs_scales.lin_vel
        self.ang_vel_scale = self.cfg.normalization.obs_scales.ang_vel
        self.dof_pos_scale = self.cfg.normalization.obs_scales.dof_pos
        self.dof_vel_scale = self.cfg.normalization.obs_scales.dof_vel
        self.command_scale = np.array(self.cfg.normalization.obs_scales.command)
        # ====== rewards ======
        self._past_targets = []
        self._past_pos = []
        self._last_traj_index = 1 # 记录机器人经过的上一个目标点
        self.episode_sums = {name: 0. for name in self.reward_scales.keys()}
        # ====== ternimal ======
        self._finish = False
        self._unhealthy = False
        self._timeout = False
        # ====== info ======
        self._info = {}
        self._info["reward"] = {}
        self._info["reward"]["cur_reward"] = {}
        self._info["reward"]["episode_sums"] = {}
        self._info["states"] = []
        # ====== episode init ======
        self._reward = 0
        self._is_terminal = False
        self._observation = 0

# *************************************Simulation*************************************************

    def prior_physics(self, action):
        """在 simulation 前进行预处理
        看下来主要是对上一轮的数据进行备份
        """
        self._last_action = self._action # 这里没有必要进行深拷贝，因为后面 _action 被重新指向了，不会干扰到 _last_action
        self._action = action
        self._last_qvel = np.array(self.data.qvel)
        self._last_qpos = np.array(self.data.qpos)
        self._last_base_pos = self._base_pos
        self._last_joint_pos = self._last_qpos.copy()[-self.cfg.env.num_actions:]
        self._last_joint_vel = self._last_qvel.copy()[-self.cfg.env.num_actions:]
        self._last_joint_torque = np.array(self.data.ctrl[:self.cfg.env.num_actions])

    def post_physics(self):
        """在一轮仿真后的处理
        """
        self.step_counter += 1
        self._qpos = np.array(self.data.qpos)
        self._qvel = np.array(self.data.qvel)

        self._joint_pos = self._qpos.copy()[-self.cfg.env.num_actions:]
        self._joint_vel = self._qvel.copy()[-self.cfg.env.num_actions:]
        self._joint_torque = np.array(self.data.ctrl[:self.cfg.env.num_actions])

        self._base_quat = np.array([self._qpos[4], self._qpos[5], self._qpos[6], self._qpos[3]])
        self._base_rotate = R.from_quat(self._base_quat)
        self._rotation_mat_b2w = self._base_rotate.as_matrix()
        self._rotation_mat_w2b = self._rotation_mat_b2w.T.copy()
        self._base_pos = self._qpos.copy()[:3]

        self._base_lin_vel_b = self._rotation_mat_w2b @ self._qvel[:3]
        self._base_ang_vel_b = self._qvel.copy()[3:6]
        self._base_ang_vel_w = self._rotation_mat_w2b.T @ self._qvel[3:6]

        self._pos_his[:-1, :] = self._pos_his[1:, :] # [0:-1] 从第一行到倒数第二行结束
        self._pos_his[-1] = self._base_pos.copy() # 这里对 his 进行了一次滚动式更新
        self._vel_his[:-1, :] = self._vel_his[1:, :]
        self._vel_his[-1] = self._qvel.copy()[:3]

        self._info["states"].append(self.data.qpos)

    def _do_simulation(self, action, n_frames, actuator_index=None): # 最后一个参数用以确定电机顺序是否需要重排
        """检查输入并与 mujoco 交互进行仿真
        观察下来它作为 step 的工具函数，所以我在函数命名前加了 _
        """
        if self.pause:
            return 0
        if action.shape[0] != self.cfg.env.num_actions:
            raise ValueError(f"Action dimension is wrong. Expected: {self.cfg.env.num_actions}, got: {action.shape[0]}")
        if actuator_index is None:
            self.data.ctrl = action.tolist()
        else:
            action_temp = np.array(self.data.ctrl)
            action_temp[actuator_index] = action
            self.data.ctrl = action_temp.tolist()

        for _ in range(n_frames):
            mj.mj_step(self.model, self.data)



# *************************************Ustensiles*************************************************
    def _get_noise_scale_vec(self):
        """
        这里运用了 numpy 的广播特性，所以可以用 python 标量对 array 切片进行赋值
        """
        noise_vec = np.zeros(48)
        if self.cfg.terrain.measure_heights:
            noise_vec = np.zeros(self.cfg.env.num_observations)
        self.add_noise = self.cfg.noise.add_noise
        if self.add_noise:
            noise_scales = self.cfg.noise.noise_scales
            noise_level = self.cfg.noise.noise_level
            noise_vec[:3] = noise_scales.lin_vel * noise_level * self.cfg.normalization.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.cfg.normalization.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.gravity * noise_level
            noise_vec[9:12] = 0.  # commands
            noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.dof_pos_scale
            noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.dof_vel_scale
            noise_vec[36:48] = 0.  # previous actions
            if self.cfg.terrain.measure_heights:
                noise_vec[48:self.cfg.env.num_observations] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def push_robots(self):
        """疑问：通过突然加速度来 push 机器人，合理吗
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.data.qvel[0:2] += np.random.uniform(-max_vel, max_vel, 2)