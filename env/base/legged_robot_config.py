# 在这里定义所有 rl 训练过程中的超参
# 还有待修改，mujoco 中所需的参数与 isaac 中并不一样

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 235
        num_privileged_obs = None    # if not None a privilege_obs_buf will be returned by step() (critic obs for asymmetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.             # not used with heightfields/trimeshes  疑问：这个参数是干什么用的
        send_timeouts = True         # send time out information to the algorithm
        episode_length_s = 20        # episode length in seconds
        num_history_obs = 10

    class terrain:
        curriculum = True
        mesh_type = 'trimesh'        # "heightfield" # none, plane, heightfield or trimesh # 疑问：什么
        horizontal_scale = 0.1   # [m]
        vertical_scale = 0.005   # [m]
        border_size = 25         # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1m x 1.6m rectangle (without center line) # 疑问：这个点是干什么用的
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False             # select a unique terrain type and pass all arguments
        terrain_kwargs = None        # Dict of arguments for selected terrain
        max_init_terrain_level = 5   # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10                 # number of terrain rows (levels)
        num_cols = 20                # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_threshold = 0.75       # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4             # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error) # 疑问：命令数目有什么用
        resampling_time = 10.        # time before command are changed[s] # 疑问
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]  # 疑问：heading mode 是什么

    # 这里的初始位置应该只影响后续的计算，而并不会影响模型本身
    # 模型本身的数据是由 xml 文件决定的
    class init_state:
        pos = [0.0, 0.0, 1.]         # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]   # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]    # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]    # x,y,z [rad/s]
        default_joint_angles = {     # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        
    class dof_pos_limits:
        dof_pos_limits = [[0., 0.] for i in range(12)]

    class default_dof_pos:
        default_dof_pos = [0.0 for i in range(12)]
        
    class control:
        control_type = 'P'           # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT # 疑问：通俗的来说是不是与控制频率挂钩，这个数越大代表你的控制频率越低
        decimation = 2
        kp = 0.6
        kv = 0.08

    class asset:                             # 疑问：感觉这里面的很多都需要解释
        xml_path = ""
        name = "legged_robot"                # actor name
        body_name = []                   # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True         # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False                # fixe the base of the robot
        default_dof_drive_mode = 3           # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort) 疑问：effort 是力矩控制吗
        self_collisions = 0                  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True       # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  .0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True     # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25            # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.          # percentage of urdf limits, values above this limit are penalized # 疑问：所谓的软限制就是指超过这个的值会被惩罚，所以这个值会尽量少的出现吗
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.         # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            command = [lin_vel, lin_vel, ang_vel]
        clip_observations = 100.
        clip_actions = 100.
        clip_torques = 6.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    # 应该是 isaac 的一些配置
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        pause = False
        overlay = {}

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

# 感觉划分的其实不是很合理，里面也不全是 PPO 的内容
class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5      # 疑问：这个 epoch 数是什么意思
        num_mini_batches = 4         # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3        # 5.e-4
        schedule = 'adaptive'        # could be adaptive, fixed 疑问：什么意思
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        empirical_normalization = False # 这个功能目前未知

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 5       # per iteration 疑问：什么意思这个数字
        max_iterations = 1500        # number of policy updates

        # logging
        save_interval = 10          # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False               # 疑问：resume 什么？
        load_run = -1                # -1 = last run
        checkpoint = -1              # -1 = last saved model
        resume_path = None           # updated from load_run and chkpt