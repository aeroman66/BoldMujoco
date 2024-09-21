import  torch
from torch.utils.tensorboard import SummaryWriter

class OnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device='cpu') -> None:
        self.device = device
        self.env = env
        self.cfg = train_cfg
        self.writer = SummaryWriter(log_dir=log_dir)
        self.score_dict = {}
        obs = self.env.get_observations()
        num_obs = obs.shape[0]
        num_obs_critic = num_obs
        