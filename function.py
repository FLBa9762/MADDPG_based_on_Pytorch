import torch
import torch.nn.functional as F
from gym import spaces
from multiagent.multi_discrete import MultiDiscrete



class PolicyDistribution(object):
    def sample(self):
        """Sampling from the policy distribution."""
        raise NotImplementedError

    def entropy(self):
        """The entropy of the policy distribution."""
        raise NotImplementedError

    def kl(self, other):
        """The KL-divergence between self policy distributions and other."""
        raise NotImplementedError

    def logp(self, actions):
        """The log-probabilities of the actions in this policy distribution."""
        raise NotImplementedError


class CategoricalDistribution(PolicyDistribution):
    """Categorical distribution for discrete action spaces."""

    def __init__(self, logits):
        """
        Args:
            logits: A float32 tensor with shape [BATCH_SIZE, NUM_ACTIONS] of unnormalized policy logits
        """
        assert len(logits.shape) == 2
        self.logits = logits

    def sample(self):
        """
        Returns:
            sample_action: An int64 tensor with shape [BATCH_SIZE] of multinomial sampling ids.
                           Each value in sample_action is in [0, NUM_ACTIOINS - 1]
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns:
            entropy: A float32 tensor with shape [BATCH_SIZE] of entropy of self policy distribution.
        """
        logits = self.logits - torch.max(self.logits, dim=1)
        e_logits = torch.exp(logits)
        z = torch.sum(e_logits, dim=1)
        prob = e_logits / z
        entropy = -1.0 * torch.sum(prob * (logits - torch.log(z)), dim=1)

        return entropy

    def logp(self, actions, eps=1e-6):
        """
        Args:
            actions: An int64 tensor with shape [BATCH_SIZE]
            eps: A small float constant that avoids underflows when computing the log probability

        Returns:
            actions_log_prob: A float32 tensor with shape [BATCH_SIZE]
        """
        assert len(actions.shape) == 1

        logits = self.logits - torch.max(self.logits, dim=1)
        e_logits = torch.exp(logits)
        z = torch.sum(e_logits, dim=1)
        prob = e_logits / z

        actions = torch.unsqueeze(actions, dim=1)
        actions_onehot = F.one_hot(actions, prob.shape[1])
        actions_onehot = torch.cast(actions_onehot, dtype='float32')
        actions_prob = torch.sum(prob * actions_onehot, dim=1)

        actions_prob = actions_prob + eps
        actions_log_prob = torch.log(actions_prob)

        return actions_log_prob

    def kl(self, other):
        """
        Args:
            other: object of CategoricalDistribution

        Returns:
            kl: A float32 tensor with shape [BATCH_SIZE]
        """
        assert isinstance(other, CategoricalDistribution)

        logits = self.logits - torch.max(self.logits, dim=1)
        other_logits = other.logits - torch.max(other.logits, dim=1)

        e_logits = torch.exp(logits)
        other_e_logits = torch.exp(other_logits)

        z = torch.sum(e_logits, dim=1)
        other_z = torch.sum(other_e_logits, dim=1)

        prob = e_logits / z
        kl = torch.sum(
            prob * (logits - torch.log(z) - other_logits + torch.log(other_z)),
            dim=1)
        return kl

class SoftCategoricalDistribution(CategoricalDistribution):
    """Categorical distribution with noise for discrete action spaces"""

    def __init__(self, logits):
        """
        Args:
            logits: A float32 tensor with shape [BATCH_SIZE, NUM_ACTIONS] of unnormalized policy logits
        """
        self.logits = logits
        super(SoftCategoricalDistribution, self).__init__(logits)

    def sample(self):
        """
        Returns:
            sample_action: An int64 tensor with shape [BATCH_SIZE, NUM_ACTIOINS] of sample action,
                           with noise to keep the target close to the original action.
        """
        eps = 1e-4
        uniform = torch.rand_like(self.logits)
        soft_uniform = torch.log(-1.0 * torch.log(uniform))
        return F.softmax(self.logits - soft_uniform, dim=-1)


class SoftMultiCategoricalDistribution(PolicyDistribution):
    """Categorical distribution with noise for MultiDiscrete action spaces."""

    def __init__(self, logits, low, high):
        """
        Args:
            logits: A float32 tensor with shape [BATCH_SIZE, LEN_MultiDiscrete, NUM_ACTIONS] of unnormalized policy logits
            low: lower bounds of sample action
            high: Upper bounds of action
        """
        self.logits = logits
        self.low = low
        self.high = high
        self.categoricals = list(
            map(
                SoftCategoricalDistribution,
                torch.split(
                    logits,
                    split_size_or_sections=list(high - low + 1),
                    dim=len(logits.shape) - 1)))

    def sample(self):
        """
        Returns:
            sample_action: An int64 tensor with shape [BATCH_SIZE, NUM_ACTIOINS] of sample action,
                           with noise to keep the target close to the original action.
        """
        cate_list = []
        for i in range(len(self.categoricals)):
            cate_list.append(self.low[i] + self.categoricals[i].sample())
        return torch.cat(cate_list, dim=-1)

    def layers_add_n(self, input_list):
        """
        Adds all input tensors element-wise, can replace tf.add_n
        """
        assert len(input_list) >= 1
        res = input_list[0]
        for i in range(1, len(input_list)):
            res = torch.add(res, input_list[i])
        return res

    def entropy(self):
        """
        Returns:
            entropy: A float32 tensor with shape [BATCH_SIZE] of entropy of self policy distribution.
        """
        return self.layers_add_n([p.entropy() for p in self.categoricals])

    def kl(self, other):
        """
        Args:
            other: object of SoftCategoricalDistribution

        Returns:
            kl: A float32 tensor with shape [BATCH_SIZE]
        """
        return self.layers_add_n(
            [p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])


def get_shape(input_space):
    if (isinstance(input_space, spaces.Box)):
        if(len(input_space.shape) == 1):
            return input_space.shape[0]
        else:
            return input_space.shape

    elif(isinstance(input_space, spaces.Discrete)):
        return input_space.n

    elif(isinstance(input_space, MultiDiscrete)):
        return sum(input_space.high - input_space.low + 1)

    else:
        print('[Error] shape is {}, not Box or Discrete or MultiDiscrete'.
              format(input_space.shape))
        raise NotImplementedError

def list_concat(obs, act):
    for i, num in enumerate(obs):
        if i == 0:
            critic_all = obs[i]
        else:
            critic_all = torch.cat((critic_all, obs[i]), 1)
    for i, num in enumerate(act):
        critic_all = torch.cat((critic_all, act[i]), 1)

    return critic_all

def SoftPDistribution(logits, act_space):
    if (hasattr(act_space, 'n')):   # 判断对象 action_space 是否含有属性 'n'
        return SoftCategoricalDistribution(logits)
    # is instance of multiagent.multi_discrete.MultiDiscrete
    elif (hasattr(act_space, 'num_discrete_space')):
        return SoftMultiCategoricalDistribution(logits, act_space.low,
                                                act_space.high)
    else:
        raise AssertionError("act_space must be instance of \
            gym.spaces.Discrete or multiagent.multi_discrete.MultiDiscrete")