import numpy as np
import torch

class Reward:
    def __init__(self) -> None:
        tracking_sigma_lin_vel = 0.25
        tracking_sigma_ang_vel = 0.25
        reference_height = 0.34
        body = [4,7,10,13]
        pen_body = [3,4,7,8,11,12,15,16]  # Right is 5 9  13 17
        right_body = [5,9,13,17]
        foot = [13, 22, 32, 42]
        height_des = 0.45
        height_threshold = 0.15
        reward_joint_acc_coef = -0.05
        reward_joint_vel_coef = -0.5
        pos_threshold = 0.65

        self.tracking_sigma_lin_vel = tracking_sigma_lin_vel
        self.tracking_sigma_ang_vel = tracking_sigma_ang_vel
        self.reference_height = reference_height
        self.body = body
        self.pen_body = pen_body
        self.right_body = right_body
        self.height_des = height_des
        self.height_threshold = height_threshold
        self.foot = foot
        self._reward_joint_acc_coef = reward_joint_acc_coef
        self._reward_joint_vel_coef = reward_joint_vel_coef
        self.pos_threshold = pos_threshold

    def _reward_track_lin_vel(self):
        command = self.command
        lin_vel_error = np.sum(np.square(self.base_lin_vel[:2] - command[:2]))
        return np.exp(-lin_vel_error / self.tracking_sigma_lin_vel)
    
    def _reward_track_ang_vel(self):
        command = self.command
        ang_vel_error = np.sum(np.square(self.base_ang_vel[:2] - command[:2]))
        return np.exp(-ang_vel_error / self.tracking_sigma_ang_vel)
    
    def _reward_lin_vel_z(self):
        return np.square(self._qvel[2])
    
    def _reward_ang_vel_xy(self):
        return np.sum(np.square(self._nase_ang_vel[:2]))
    
    def _reward_orientation(self):
        gravity = [0., 0., -1.0]
        gravity_proj = self._rotation_mat_w2b @ gravity
        return np.sum(np.square(gravity_proj[:2]))
    
    def _reward_base_height(self):
        body_height = self._qpos[2]
        return np.square(body_height - self.reference_height)
    
    def _reward_torques(self):
        """注意此处torque只是电机输入
        """
        return np.linalg.norm(self._joint_torque)
    
    def _reward_dof_acc(self):
        """Penalize dof accelerations
        """
        return np.sum(np.square((self._last_joint_vel - self._joint_vel) / self.dt))
    
    def _reward_action_rate(self):
        """Penalize changes in actions 注此处的action是目标位置，根据pd控制原理算出的输入即为torque上两个的
        """
        return np.sum(np.square(self._last_action - self._action))
    
    def _reward_collision(self):
        """go1 penalize with geom, first compute every geoms total contact force
        if thigh and calf contact, both geom yield con_force, both count
        """
        wrong_list=[]
        wrong_body_num = 0
        for i, c in enumerate(self.data.contact): # 该函数同时返回索引和值
            # print('contact : ', i, 'body : ', self.model.geom_bodyid[self.data.contact[i].geom1],self.model.geom_bodyid[self.data.contact[i].geom2],
            #       self.data.xpos[self.model.geom_bodyid[self.data.contact[i].geom2]][0], self.data.xpos[self.model.geom_bodyid[self.data.contact[i].geom2]][1],
            #       self.data.xpos[self.model.geom_bodyid[self.data.contact[i].geom2]][2])
            # 分别检查这碰撞中的一对几何体是否在惩罚项中
            if (self.model.geom_bodyid[self.data.contact[i].geom1] in self.pen_body):
                wrong_list.append(self.model.geom_bodyid[self.data.contact[i].geom1])
            if (self.model.geom_bodyid[self.data.contact[i].geom2] in self.pen_body):
                wrong_list.append(self.model.geom_bodyid[self.data.contact[i].geom2])
            # if (self.data.contact[i].geom1 == 0 and self.data.contact[i].geom2 in ):
            #     wrong_body_num -= 1
        unique_list = list(set(wrong_list)) # 需要进行去重操作
        # print(unique_list)
        wrong_body_num = len(unique_list)
        # print("wrong_body_num", wrong_body_num)
        return wrong_body_num
    
    # 这个写错了啊，跟碰撞惩罚函数一模一样
    # def _reward_contact_time(self):
    #     # go1 penalize with geom, first compute every geoms total contact force
    #     # if thigh and calf contact, both geom yield con_force, both count
    #     con_geom2 = []
    #     for i, c in enumerate(self.data.contact):
    #         if (self.data.contact[i].geom1 == 0 and self.model.geom_bodyid[self.data.contact[i].geom2] in self.right_body):
    #             con_geom2.append(self.data.contact[i].geom2)
    #     con_list = [item in con_geom2 for item in self.right_body]
    #     con_filt = np.array(con_list) * np.array(self._last_con_list)
    #     self._last_con_list = con_list

    #     wrong_list=[]
    #     wrong_body_num = 0
    #     for i, c in enumerate(self.data.contact):
    #         print('contact : ', i, 'body : ', self.model.geom_bodyid[self.data.contact[i].geom1],self.model.geom_bodyid[self.data.contact[i].geom2],
    #               self.data.xpos[self.model.geom_bodyid[self.data.contact[i].geom2]][0], self.data.xpos[self.model.geom_bodyid[self.data.contact[i].geom2]][1],
    #               self.data.xpos[self.model.geom_bodyid[self.data.contact[i].geom2]][2])
    #         if (self.model.geom_bodyid[self.data.contact[i].geom1] in pen_body):
    #             wrong_list.append(self.model.geom_bodyid[self.data.contact[i].geom1])
    #         if (self.model.geom_bodyid[self.data.contact[i].geom2] in pen_body):
    #             wrong_list.append(self.model.geom_bodyid[self.data.contact[i].geom2])
    #         # if (self.data.contact[i].geom1 == 0 and self.data.contact[i].geom2 in ):
    #         #     wrong_body_num -= 1
    #     unique_list = list(set(wrong_list))
    #     # print(unique_list)
    #     wrong_body_num = len(unique_list)
    #     # print("wrong_body_num", wrong_body_num)
    #     return wrong_body_num

    def _reward_dof_pos_limits(self):
        """Penalize dof positions too close to the limit
        """
        out_of_limits = -(self._joint_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self._joint_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return np.sum(out_of_limits, axis=0)
    
    def _reward_track_target_quat(self):
        real_pos = np.concatenate([self._qpos[0:2], self._qpos[3:7]])
        track_pos = np.concatenate([self.init_qpos[0:2], self.init_qpos[3:7]])
        diff_squared = (real_pos - track_pos) ** 2
        squared_sum = np.sum(diff_squared)
        # print(squared_sum)
        return squared_sum

    def _reward_track_target_joint(self):
        real_pos = self._qpos[7:]
        track_pos = self.init_qpos[7:]
        diff_squared = (real_pos - track_pos) ** 2
        squared_sum = np.sum(diff_squared)
        # print(squared_sum)
        return squared_sum


    def _reward_quat_vel(self):
        real_quat_vel_2 = self._base_ang_vel_w ** 2
        return np.sum(real_quat_vel_2)


    def _reward_keep_time(self):
        return (self._keep_time)
    
    def _reward_move_forward(self):
        base_pos_x = self._base_pos[0]
        last_base_pos_x = self._last_base_pos[0]
        return (base_pos_x - last_base_pos_x)
    
    def _reward_pose(self):
        """奖励机身高度保持在一定范围内
        不是很确定和前面的高度惩罚项的区别在哪里
        """
        height_cur = self._base_pos[2]
        height_error = np.abs(self.height_des - height_cur)
        
        # 卡在0.3到0.6之间，中点是0.45,门限是0.15,正常是0.35
        height_error_tensor = torch.from_numpy(np.array([height_error]))
        height_norm_error = torch.relu(torch.abs(height_error_tensor) - self.height_threshold)
        # print(height_norm_error.item())
        return height_norm_error.item() # 转化为张量是为了进行激活操作，而 item() 只能对单元素张量进行操作
    
    def _reward_action(self):
        """单纯是不想让动作幅度起飞
        """
        # print(np.sum(np.square(self._action)))
        return np.sum(np.square(self._action))

    def _reward_contact_cost(self):
        """为了不让触地力过大
        """
        raw_contact_forces = np.array(self.data.cfrc_ext)
        contact_forces = np.clip(raw_contact_forces, -100, +100) # 注意这里进行了截断操作
        return np.sum(np.square(contact_forces))
    
    def _reward_contact_pair(self):
        foot_right = []
        for i, c in enumerate(self.data.contact):
            if (self.data.contact[i].geom1 == 0 and self.data.contact[i].geom2 in self.foot): # 编号为 0 的 geom 代表地面
                foot_right.append(1)
            else:
                foot_right.append(-1)
        length = len(foot_right)
        if length == 0:
            return 0
        else:
            return (np.sum(np.array(foot_right)) / length)
        
    def _reward_joint_acc(self):
        joint_acc = (self._joint_vel - self._last_joint_vel) / self.dt
        return np.linalg.norm(joint_acc)


    def _reward_joint_vel(self):
        return np.linalg.norm(self._joint_vel)


    def _reward_action_change(self):
        action_change = np.abs(self._action - self._last_action) / self.dt
        return np.linalg.norm(action_change)


    def _reward_joint_range(self):
        """惩罚关节角度超过固定阈值范围
        :return: exp(joint_pos - threshold) - 1
        """
        joint_hip = np.array([self._joint_pos[0],self._joint_pos[3],self._joint_pos[6],self._joint_pos[9]])
        hip_tensor = torch.from_numpy(joint_hip)
        joint_pos_tensor = torch.from_numpy(self._joint_pos - self.init_qpos[7:])
        joint_over_range = torch.relu(torch.abs(hip_tensor) - self.pos_threshold)
        over_range_punish = (torch.exp(joint_over_range) - 1.0).sum()
        # over_range_punish = joint_over_range.sum()
        return over_range_punish.item()
    
    def _reward_joint_pos(self):
        """
        :return:???????????????????????????
        """
        return (abs(self._joint_pos).sum() - 2.0) / (self.curriculum + 1)
    
    # def _reward_hip_inward(self):
    #     """*未完成(walk dog no need)
    #     惩罚左右髋关节同时向内的动作
    #     :return:
    #     """
    #     hip_index = [0, 3, 6, 9, 12, 15]
    #     hip_joint_pos = self._joint_pos[hip_index]
    #     inward_direction = np.array([-1, -1, -1, 1, 1, 1])  # *方向错误

    # def _reward_contact_area(self):
    #     """
    #     *未完成(walk dog no need)
    #     鼓励机器人增大触地点的面积
    #     :return:
    #     """
    #     threshold = 1.5  # 髋关节都为0时的小腿y坐标绝对值之和
    #     shank_index = [4, 8, 11, 15, 19, 22]
    #     shank_xpos_xy_b = self.data.xpos[shank_index, 0:2] - self._base_pos[0:2]
    #     contact_area = np.abs(shank_xpos_xy_b)[:, 1].sum()
    #     return -np.exp(- contact_area + threshold) + 1

    def _reward_heading(self):
        targets_vector = (self._observe_targets - self._base_pos)
        targets_vector[:, 2] = 0
        targets_direction = targets_vector / np.linalg.norm(targets_vector, ord=2, axis=1, keepdims=True)
        targets_dire_avg = np.mean(targets_direction, axis=0)
        heading = -self._rotation_mat_b2w[:, 0]
        threshold = np.cos(np.deg2rad(45))
        heading_reward = np.exp(np.dot(heading, targets_dire_avg)) - np.exp(threshold)
        return heading_reward

    def _reward_time_consume(self):
        return 2.0 / (self.curriculum + 1)

    def _reward_termination(self):
        if self._unhealthy:
            return -20.
        if self._timeout:
            return -6.
        if self._finish:
            return 100.
        return 0.