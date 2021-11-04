import torch
import numpy as np
from replay_memory import ReplayMemory
from para import parse_args
import os


class Agent():
    def __init__(self,
                 algorithm,
                 agent_index=None,
                 obs_dim_n=None,
                 act_dim_n=None,
                 batch_size=None,
                 ):
        assert isinstance(agent_index, int)
        assert isinstance(obs_dim_n, list)
        assert isinstance(act_dim_n, list)
        assert isinstance(batch_size, int)

        self.alg = algorithm
        self.agent_index = agent_index
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.batch_size = batch_size

        self.memory_size = int(1e5)
        self.min_memory_size = batch_size * 25
        self.rpm = ReplayMemory(self.memory_size,
                                self.obs_dim_n[agent_index],
                                self.act_dim_n[agent_index])
        self.global_train_step = 0
        self.n = len(act_dim_n)  # 智能体数目
        self.alg.sync_target(decay=0)   # 初始化复制网络参数
        self.show_begin = True

    def predict(self, obs, arglist, use_target_model=False):    # 无初始值参数只可以在有初始值参数之前
        device = arglist.device
        obs = torch.from_numpy(obs.reshape(1, -1)).to(device, torch.float)
        # TODO 这个act梯度的切断时机，PARL里面从模型出来时候已经没有梯度了
        act = self.alg.predict(obs, use_target_model=use_target_model)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def learn(self, agents, arg):        # 状态采样，计算 q_target, 训练
        device = arg.device

        self.global_train_step += 1

        # only update parameter every 100 steps
        if self.global_train_step % 100 != 0:
            return 0.0

        if self.rpm.size() <= self.min_memory_size:
            return 0.0
        if self.show_begin:
            print("*****************Begin Learning******************")
            self.show_begin = False

        batch_obs_n = []
        batch_act_n = []
        batch_obs_next_n = []

        rpm_sample_index = self.rpm.make_index(self.batch_size)
        for i in range(self.n):
            batch_obs, batch_act, _, batch_obs_next, _\
                = agents[i].rpm.sample_batch_by_index(rpm_sample_index)
            batch_obs_n.append(batch_obs)
            batch_act_n.append(batch_act)
            batch_obs_next_n.append(batch_obs_next)

        _, _, batch_rew, _, batch_isOver = self.rpm.sample_batch_by_index(rpm_sample_index)
        batch_obs_n = [torch.from_numpy(obs).to(device, torch.float) for obs in batch_obs_n]

        batch_act_n = [torch.from_numpy(act).to(device, torch.float) for act in batch_act_n]

        batch_rew = torch.from_numpy(batch_rew).to(device, torch.float)
        batch_isover = torch.from_numpy(batch_isOver).to(device, torch.float)

        target_act_next_n = []
        batch_obs_next_n = [
            torch.from_numpy(obs).to(device, torch.float) for obs in batch_obs_next_n
        ]

        for i in range(self.n):
            target_act_next = agents[i].alg.predict(
                batch_obs_next_n[i], use_target_model=True
            )
            target_act_next = target_act_next.detach()
            target_act_next_n.append(target_act_next)
        target_q_next = self.alg.Q_value(batch_obs_next_n, target_act_next_n, use_target_model=True)
        target_q = batch_rew + self.alg.gamma * (1.0 - batch_isover) * target_q_next.detach()

        critic_cost = self.alg.learn(batch_obs_n, batch_act_n, target_q)
        critic_cost = critic_cost.cpu().detach().numpy()
        # print("critic_cost:{}".format(critic_cost))
        return critic_cost

    def add_experience(self, obs, act, reward, next_obs, terminal):
        self.rpm.append(obs, act, reward, next_obs, terminal)

    def restore(self, save_path, model=None, map_location=None):
        if model is None:
            model = self.alg.model
        checkpoint = torch.load(save_path, map_location=map_location)
        model.load_state_dict(checkpoint)

    def save(self, save_path, model=None):
        if model is None:
            model = self.alg.model
        sep = os.sep
        dirname = sep.join(save_path.split(sep)[:-1])
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(model.state_dict(), save_path)



