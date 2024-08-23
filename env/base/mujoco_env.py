# 主要负责创建 mujoco 仿真环境
# 与 legged_gym 采用了不同的思路，因为 mujoco 的环境创建需要一个具体的 xml 模型
# 所以这个训练从这里开始就需要与具体的机器人配置文件绑定了

# from legged_robot_config import LeggedRobotCfg # 这里不一定需要传入这个配置，可以随机器人型号而定
import mujoco as mj
import mujoco.glfw as glfw
import numpy as np # 因为 mujoco 应该不支持 GPU 计算，所以这里大多用的是 numpy 库
import os

class MujocoEnv(): # 疑问：这里写上的括号有任何作用吗
    def __init__(self, cfg) -> None: # cfg 的具体传入类型由机器人型号决定
        self.frame_skip = cfg.sim.frame_skip
        self.xml_path = cfg.asset.xml_path
        self.body_name = cfg.asset.body_name
        self.overlay = cfg.sim.overlay

        # mujoco data structure
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

        self.init_qpos = np.array(self.data.qpos)
        self.init_qvel = np.array(self.data.qvel)

        # For callback functions 都是为了仿真 UI 服务的
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        self.init_window()
        self.create_overlay(self.model, self.data)

    @property
    def dt(self):
        """
        有了这个装饰器，我调用 dt 的时候可以直接向我返回这个 return 结果，就像一个属性一样
        有时我们这么做是因为我们希望对 原数值进行一些计算再返回
        """
        return self.model.opt.timestep * self.frame_skip

    # 都是为 mujoco UI 编写的函数
    def do_simulation(self, ctrl, n_frames, actuator_index=None):
        if self.pause:
            return 0
        
        if ctrl.shape[0] != self.action_space["action_num"]:
            raise ValueError("Action dimension mismatch")

        if actuator_index == None:
            self.data.ctrl = ctrl.tolist()
        else:
            ctrl_temp = np.array(self.data.ctrl)
            ctrl_temp[actuator_index] = ctrl
            self.data.ctrl = ctrl_temp.tolist()

        for _ in range(n_frames):
            mj.mj_step(self.model, self.data)

    def init_glfw(self):
        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

    def init_visual(self):
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        # self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
        self.opt.frame = 1

        self.model.vis.map.force = 0.01

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

    def init_window(self):
        self.init_glfw()
        self.init_visual()

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)

    def render(self):
        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        self.create_overlay(self.model, self.data)

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                           mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # 渲染 UI 列表
        for gridpos, [t1, t2] in self.overlay.items():
            mj.mjr_overlay(
                mj.mjtFontScale.mjFONTSCALE_150,
                gridpos,
                viewport,
                t1,
                t2,
                self.context)

        # clear overlay
        self.overlay.clear()

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)
        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
        if act == glfw.PRESS and key == glfw.KEY_ESCAPE:
            print("sim end")
            glfw.set_window_should_close(self.window, True)
        if act == glfw.PRESS and key == glfw.KEY_V:
            self.pause = not self.pause
            if self.pause:
                print("pause")
            else:
                print("resume")

    def mouse_button(self, window, button, act, mods):
        # update button state

        self.button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save

        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(
            window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(
            window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx / height,
                          dy / height, self.scene, self.cam)

    def scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                          yoffset, self.scene, self.cam)
        
    def add_overlay(self, gridpos, text1, text2):
        if gridpos not in self.overlay:
            self.overlay[gridpos] = ["", ""]
        self.overlay[gridpos][0] += text1 + "\n"
        self.overlay[gridpos][1] += text2 + "\n"

    def create_overlay(self, model, data):
        """
        创建 UI 列表
        """
        topleft = mj.mjtGridPos.mjGRID_TOPLEFT
        topright = mj.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mj.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mj.mjtGridPos.mjGRID_BOTTOMRIGHT

        self.add_overlay(
            bottomleft,
            "Pause",
            'v',)
        self.add_overlay(
            bottomleft,
            "Reset",
            'Backspace',)
        self.add_overlay(
            bottomleft,
            "Time",
            '%.2f' % self.data.time,)

    # 创建一些工具函数
    def _set_action_space(self):
        """
        从 mujoco 模型中获取了各种关于动作命令的信息
        并将其存储与一个字典中
        """
        bounds = self.model.acactuator_ctrlrange.copy().astype(np.float32)
        self.action_space = {}
        self.action_space["bounds"] = bounds
        self.action_space["action_num"] = bounds.shape[0]
        self.action_space["low"] = bounds[:, 0]
        self.action_space["high"] = bounds[:, 1]
        actions = np.zeros(bounds.shape[0])

    def _set_observation_space(self, observation):
        """
        没有直接对 observation 数据进行处理
        而是利用这部分的 observation 对字典进行了一个初始化，来获取一些想要知道的关于 obs 的数据
        """
        self.observation_space = {}
        self.observation_space["low"] = np.full(observation.shape[0], -float("inf"), dtype=np.float32)
        self.observation_space["high"] = np.full(observation.shape[0], float("inf"), dtype=np.float32)
        self.observation_space["observation_num"] = observation.shape[0]
        observation = np.zeros(observation.shape[0])

    def _init_act_obs(self):
        """
        同时对 actions 和 observation 进行初始化
        """
        # self.reset_model()
        self._set_action_space()
        observation, _reward, done, _info = self.step(np.zeros(self.action_space["action_num"]))
        self._set_observation_space(observation)
        
    # 以下是一些真正基础的与训练相关的功能函数
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, logging, and sometimes learning)
        """
        raise NotImplementedError
    
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError
    
    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def reset_model(self, return_info:bool = False):
        mj.mj_resetData(self.model, self.data) # 将 mujoco 的数据进行重置
        ob = self.reset_model()
        if not return_info: # 这个 not 加的很迷惑
            return ob
        else:
            return ob, {}   # 这个字典是什么额外信息
        
    def set_state(self, qpos, qvel, qpos_index=None, qvel_index=None):
        assert qpos.shape[0] == self.model.nq and qvel.shape[0] == self.model.nv
        self.data.qpos = qpos.tolist()
        self.data.qvel = qvel.tolist()
        mj.mj_forward(self.model, self.data) # 这个 forward 应该和 step 的功能不一样吧