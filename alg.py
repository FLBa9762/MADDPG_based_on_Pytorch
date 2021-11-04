import torch
from copy import deepcopy
from function import *




class MADDPG():
    def __init__(self, agent_index, model, actor_lr, critic_lr, act_space, gamma, tau):

        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.agent_index = agent_index
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.act_space = act_space
        self.gamma = gamma
        self.tau = tau

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.update_target_tag = 0
        self.actor_optimizer = torch.optim.Adam(lr=self.actor_lr, params=self.model.get_actor_params())
        self.critic_optimizer = torch.optim.Adam(lr=self.critic_lr, params=self.model.get_critic_params())

    def predict(self, obs, use_target_model=False):      # 求智能体动作,执行时
        if use_target_model:
            policy = self.target_model.policy(obs)
        else:
            policy = self.model.policy(obs)

        action = SoftPDistribution(
            logits=policy,
            act_space=self.act_space[self.agent_index]).sample()
        return action

    def Q_value(self, obs_n, act_n, use_target_model = False):        # Q网络
        if use_target_model:
            return self.target_model.value(obs_n, act_n)
        else:
            return self.model.value(obs_n, act_n)

    # TODO torch网络的输出格式和paddle是否有区别？
    def learn(self, obs_n, act_n, target_q):
        self.update_target_tag += 1
        actor_cost = self.actor_learn(obs_n, act_n)
        critic_cost = self.critic_learn(obs_n, act_n, target_q)

        if self.update_target_tag % 10 == 0:
            self.sync_target(self.tau)

        return critic_cost

    def actor_learn(self, obs_n, act_n):
        i = self.agent_index
        obs = obs_n[i]
        policy = self.model.policy(obs)
        sample_this_action = SoftPDistribution(logits=policy, act_space=self.act_space[self.agent_index]).sample()
        action_input_n = act_n + []
        action_input_n[i] = sample_this_action
        eval_q = self.Q_value(obs_n, action_input_n)
        act_cost = torch.mean(-1*eval_q)

        act_reg = act_cost + torch.mean(torch.square(policy))

        cost = act_cost + act_reg * 1e-3
        self.actor_optimizer.zero_grad()
        cost.backward()
        # 梯度剪辑
        torch.nn.utils.clip_grad_norm_(self.model.get_actor_params(), 0.5)
        self.actor_optimizer.step()
        return cost

    def critic_learn(self, obs_n, act_n, target_q):
        pred_q = self.Q_value(obs_n, act_n)
        cost = F.mse_loss(pred_q, target_q)

        self.critic_optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.model.get_critic_params(), 0.5)
        self.critic_optimizer.step()
        return cost

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)

