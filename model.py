
import torch
import torch.nn as nn
import torch.nn.functional as F
# from function import list_concat

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim, critic_in_dim):
        super(Model, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim)
        self.critic_model = CriticModel(critic_in_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

    def sync_weights_to(self, target_model, decay=None):        # TODO decay的实参设置
        assert not target_model is self, "cannot copy between identical model"
        assert isinstance(target_model, Model)
        assert self.__class__.__name__ == target_model.__class__.__name__, \
            "must be the same class for params syncing!"
        assert (decay >= 0 and decay <= 1)

        target_vars = dict(target_model.named_parameters())
        for name, var in self.named_parameters():
            target_vars[name].data.copy_(decay * target_vars[name].data +
                                         (1 - decay) * var.data)



class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorModel, self).__init__()
        hid1_size = 64
        hid2_size = 64
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, obs):
        hid1 = F.relu(self.fc1(obs))
        hid2 = F.relu(self.fc2(hid1))
        out = self.fc3(hid2)
        return out

    # def parameters(self, recurse: bool = True):
    #     for name, param in self.named_parameters(recurse=recurse):
    #         yield param


class CriticModel(nn.Module):
    def __init__(self, critic_in_dim):
        super(CriticModel, self).__init__()
        hid1_size = 64
        hid2_size = 64
        out_dim = 1
        self.fc1 = nn.Linear(critic_in_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, out_dim)
        # self.LReLU = nn.LeakyReLU(0.01)

    def forward(self, obs_n, act_n):
        critic_all = obs_n[0]
        for i, num in enumerate(obs_n):
            if i >= 1:
                critic_all = torch.cat((critic_all, obs_n[i]), 1)
        for i, num in enumerate(act_n):
            critic_all = torch.cat((critic_all, act_n[i]), 1)
        # print("critic_all:{}".format(critic_all))
        # print("*********")
        # inputs = list_concat(obs_n, act_n)
        hid1 = F.relu(self.fc1(critic_all))
        hid2 = F.relu(self.fc2(hid1))
        out = self.fc3(hid2)
        out = torch.squeeze(out, 1)
        # print(out)
        return out

    # def parameters(self, recurse: bool = True):
    #     for name, param in self.named_parameters(recurse=recurse):
    #         yield param


