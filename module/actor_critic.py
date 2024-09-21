import torch
import torch.nn as nn
fromtorch.distributions import Normal

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
            self,
            num_obs_actor,
            num_obs_critic,
            num_actions,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=1.0):
        super().__init__()
        activation = get_activation(activation)
        mlp_input_dim_actor = num_obs_actor
        mlp_input_dim_critic = num_obs_critic

        # ******actor******
        actor_layer = []
        actor_layer.append(nn.Linear(mlp_input_dim_actor, actor_hidden_dims[0]))
        actor_layer.append(activation)
        for layer_index in range(len(actor_hidden_dims) - 1):
            actor_layer.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
            actor_layer.append(activation)
        actor_layer.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layer)
        self.actor.float() # 将网络中的所有参数转化为 float 类型

        # ******critic******
        critic_layer = []
        critic_layer.append(nn.Linear(mlp_input_dim_critic, critic_hidden_dims[0]))
        critic_layer.append(activation)
        for layer_index in range(len(critic_hidden_dims) - 1):
            critic_layer.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
            critic_layer.append(activation)
        critic_layer.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layer)
        self.critic.float()

        print(f"Actor MLP: {self.actor}") # nn 类应该定义了 __str__ 方法，所以可以直接 print
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def init_weights(self, sequential, scales):
        """可在这选择实现定制化的参数初始化方法
        """
        raise NotImplementedError
    
    def reset(self, dones=None):
        """重置网络参数
        """
        pass

    def forward(self):
        raise NotImplementedError
    
    def update_distribution(self, obs):
        """更新动作分布
        """
        mean = self.actor(obs) # 并不会直接采用这个动作，有 std 这样模型才会做出探索
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs, **kwargs):
        """根据观测值计算动作
        """
        self.update_distribution(obs)
        action = self.distribution.sample()
        # return action.detach().cpu().numpy()
        return action
    
    def get_actions_log_prob(self, actions):
        """计算动作的对数概率
        来量化在当前策略下执行特定动作的可能性，这对于更新策略和评估动作的好坏至关重要。
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, obs):
        actions_mean = self.actor(obs) # 在推理模式下就不会进行探索的动作了
        return actions_mean
    
    def evaluate(self, critic_obs, **kwargs)
        """让 critic 网络进行评分
        """
        score = self.critic(critic_obs)
        return score

# *******************************Ustensiles************************************************

def get_activation(act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "selu":
            return nn.SELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "crelu":
            return nn.CReLU()
        elif act_name == "lrelu":
            return nn.LeakyReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        else:
            print("invalid activation function!")
            return None