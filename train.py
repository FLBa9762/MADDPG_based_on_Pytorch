import os
import time
import numpy as np
from agent import Agent
from model import Model
from alg import MADDPG
from function import *
from para import parse_args
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

# from torch.utils.tensorboard import SummaryWriter


def make_env(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def run_episode(env, agents):
    obs_n = env.reset()
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0

    while True:
        steps += 1
        action_n = [agent.predict(obs, arglist, use_target_model=False, ) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        terminal = (steps > arglist.max_step_per_episode)

        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i], next_obs_n[i], done_n[i])

        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        if done or terminal:
            break

        if arglist.show:
            time.sleep(0.1)
            env.render()

        if arglist.restore and arglist.show:
            continue

        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents, arglist)
            # if critic_loss != 0.0:
            #     SummaryWriter.add_scalar('critic_loss_%d' % i, critic_loss,
            #                        agent.global_train_step)

    return total_reward, agents_reward, steps


def train(arglist):

    env = make_env(arglist.scenario, arglist.benchmark)
    # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

    critic_dim_list = [get_shape(env.observation_space[i]) for i in range(env.n)]
    actor_dim_list = [get_shape(env.action_space[i]) for i in range(env.n)]

    critic_in_dim = sum(critic_dim_list) + sum(actor_dim_list)
    print(type(critic_in_dim))
    agents = []
    for i in range(env.n):      # 初始化 agents
        model = Model(obs_dim=critic_dim_list[i],
                      act_dim=actor_dim_list[i],
                      critic_in_dim=critic_in_dim)

        alg = MADDPG(agent_index=i,
                     model=model,
                     actor_lr=arglist.actor_lr,
                     critic_lr=arglist.critic_lr,
                     act_space=env.action_space,
                     gamma=arglist.gamma,
                     tau=arglist.tau)

        agent = Agent(algorithm=alg,
                      agent_index=i,
                      obs_dim_n=critic_dim_list,
                      act_dim_n=actor_dim_list,
                      batch_size=arglist.batch_size)
        agents.append(agent)

    total_steps = 0
    total_episodes = 0

    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n)]

    if arglist.restore:
        for i in range(len(agents)):
            model_file = arglist.model_dir + '/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} existing error'.format(model_file)
                )
            agents[i].restore(model_file)

    t_start = time.time()
    print("Starting...")

    reward_list = []
    time_list = []
    while total_episodes <= arglist.max_episode:
        start = time.perf_counter()

        ep_reward, ep_agent_rewards, steps = run_episode(env, agents)

        reward_list.append(ep_reward)
        total_steps += steps
        total_episodes += 1
        episode_rewards.append(ep_reward)
        for i in range(env.n):
            agent_rewards[i].append(ep_agent_rewards[i])

        end = time.perf_counter()
        time_use = end - start
        time_list.append(time_use)

        if total_episodes % 100 == 0:
            mean_reward = np.mean(reward_list)
            time_use_mean = np.mean(time_list)
            reward_list = []
            time_list = []

            print("episode:{}   mean_reward:{}  agent_reward:{}   mean_time:{} ".format(total_episodes, mean_reward,
                                                                                    ep_agent_rewards, time_use_mean))

    if not arglist.restore:
        model_dir = arglist.model_dir
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        for i in range(len(agents)):
            model_name = '/agent_' + str(i)
            agents[i].save(model_dir + model_name)

    if arglist.load:
        net = torch.load("F:/maddpg-master/maddpg-master/maddpg_torch/model/agent_0")
        print(net)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

