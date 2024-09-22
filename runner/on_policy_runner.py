from __future__ import annotations

import os
import sys

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

from module import ActorCritic
from storage import RolloutStorage
from algo import PPO
from env.base import LeggedRobotCfgPPO
from utils import EmpiricalNormalization

import  torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time

class OnPolicyRunner:
    def __init__(self, env, train_cfg=LeggedRobotCfgPPO, log_dir=None, device='cpu') -> None:
        self.device = device
        self.env = env
        self.cfg = train_cfg
        self.writer = SummaryWriter(log_dir=log_dir)
        self.score_dict = {}
        obs = self.env.get_observations()
        num_obs = obs.shape[0]
        num_obs_critic = num_obs

        actor_critic = ActorCritic(
            num_obs_actor=num_obs,
            num_obs_critic=num_obs_critic,
            num_actions=self.env.action_space['action_num'],
        ).to(self.device)
        
        self.agent = PPO(
            actor_critic=actor_critic,
            storage=RolloutStorage,
            num_learning_epochs=self.cfg.algorithm.num_learning_epochs,
            num_mini_batches=self.cfg.algorithm.num_mini_batches,
            gamma=self.cfg.algorithm.gamma,
            schedule=self.cfg.algorithm.schedule,
            desired_kl=self.cfg.algorithm.desired_kl,
            entropy_coef=self.cfg.algorithm.entropy_coef,
            device=self.device,
        )

        self.empirical_normalization = self.cfg.algorithm.empirical_normalization
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_obs_critic], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization

        # init storage and model
        self.agent.init_storage(
            num_envs=self.env.cfg.env.num_envs,
            num_transitions_per_env=self.cfg.runner.num_steps_per_env,
            actor_obs_shape=[num_obs],
            critic_obs_shape=[num_obs_critic],
            action_shape=[self.env.action_space['action_num']],
        )

        self.log_dir = log_dir
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0

        self.ifload = False
        if self.ifload:
            self.load('') # 这里加入模型路径

    def learn(self, num_learning_iterations : int, init_at_random_ep_len : bool = False):
        obs_array =  self.env.get_observations()
        critic_obs_array = obs_array
        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
        critic_obs_tensor = torch.FloatTensor(critic_obs_array).to(self.device)

        self.train_mode()
        ep_info = []
        rebuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(1, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.cfg.env.num_envs,dtype=torch.float, device=self.device)

        score_list = []
        score = 0.0
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for iter in range(start_iter, tot_iter): # 不是直接用 num_learning_iterations 肯定是有什么深意吧
            print("epochs : {}/{}".format(iter, tot_iter))
            start = time.time()
            with torch.inference_mode(): # 竟然用了上下文管理器，但是不知道这个管理器的功能是什么
                for step in range(self.cfg.runner.num_steps_per_env):
                    actions_tensor = self.agent.act(obs=obs_tensor, critic_obs=critic_obs_tensor)
                    actions_array = actions_tensor.cpu().numpy()
                    obs_array, reward_array, done_array, infos_array = self.env.step(actions_array)
                    self.env.render()
                    obs_array = self.obs_normalizer(obs_array)
                    critic_obs_array = obs_array

                    obs_tensor = torch.FloatTensor(obs_array).to(self.device)
                    critic_obs_tensor = torch.FloatTensor(critic_obs_array).to(self.device)
                    reward_tensor = torch.FloatTensor([reward_array]).to(self.device)
                    done_tensor = torch.BoolTensor([done_array]).to(self.device)
                    # infos_tensor = torch.FloatTensor(infos_array).to(self.device)

                    noise = (2 * torch.rand_like(obs_tensor) - 1) * self.env.noise_scale_vec # 产生噪声并加入噪声
                    obs_tensor += noise
                    critic_obs_tensor += noise

                    score += reward_tensor.sum().item()

                    # 默认 critic和actor obs是一样的,略去以下部分
                    # if "critic" in infos["observations"]:
                    #     critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    # else:
                    #     critic_obs = obs
                    # obs, critic_obs, rewards, dones = (
                    #     obs.to(self.device),
                    #     critic_obs.to(self.device),
                    #     rewards.to(self.device),
                    #     dones.to(self.device),
                    # )

                    self.agent.process_env_step(
                        rewards=reward_tensor,
                        dones=done_tensor,
                        infos=infos_array,
                    )
                    # print('done_array', done_array)
                    if done_array:
                        score_list.append(score)
                        score = 0.0
                        # 下面这段还不是很看得懂
                        keys = set(self.score_dict).union(self.env._info["reward"]["episode_sums"])
                        for key in keys:
                            try:
                                self.score_dict[key].append(self.env._info["reward"]["episode_sums"].get(key, 0))
                            except KeyError:
                                self.score_dict[key] = [self.env._info["reward"]["episode_sums"].get(key, 0)]
                        obs_array = (self.env.reset())
                        obs_array = self.obs_normalizer(obs_array)
                        critic_obs_array = obs_array
                        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
                        critic_obs_tensor = torch.FloatTensor(critic_obs_array).to(self.device)

                stop = time.time()
                collection_time = stop - start
                # Learning step
                # start = stop # 这个有什么鸡毛用啊
                self.agent.compute_returns(critic_obs_tensor)
            mean_value_loss, mean_surrogate_loss = self.agent.update()

            stop = time.time()
            learn_time = stop - start
            try:
                mean_score = sum(score_list)/len(score_list)
            except ZeroDivisionError:
                mean_score = 0.0
            self.writer.add_scalar("loss/actor_loss", mean_surrogate_loss, iter)
            self.writer.add_scalar("loss/critic_loss", mean_value_loss, iter)
            self.writer.add_scalar("command/command", self.env.command[0], iter)
            self.writer.add_scalar("score/mean_score", mean_score, iter)
            for key in self.score_dict.keys():
                if len(self.score_dict[key]) > 0:
                    mean_score_key = sum(self.score_dict[key]) / len(self.score_dict[key])
                    self.writer.add_scalar(f"score/mean_score_{key}", mean_score_key, iter)
            self.current_learning_iteration = iter
            # if self.log_dir is not None:
            #     self.log(locals())
            print(type(iter % self.cfg.runner.save_interval))
            if iter % self.cfg.runner.save_interval == 0:
                print(f"saving model at {iter}")
                self.save(os.path.join(self.log_dir, f"model_{iter}.pt"))
            ep_info.clear()
            score_list = []
            self.score_dict = {}

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

# **************************save and load*******************************
    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.agent.actor_critic.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

    def load(self, path, load_optimizer=True):
        # loaded_dict = torch.load(path)
        try:
            loaded_dict = torch.load(path, map_location=torch.device('cpu'))
            print('File loaded:)')    
        except FileNotFoundError:
            print(f"File '{path}' missing:(")
        try:
            # print(loaded_dict["model_state_dict"])
            self.agent.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
            # print('iter',loaded_dict['iter'])
        except:
            print('fail load')
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.agent.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]


# *************************modes******************************
    def train_mode(self):
        self.agent.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.agent.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()