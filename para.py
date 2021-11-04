import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--num-adversaries", type=int, default=0, help="num of adversaries")
    parser.add_argument("--actor-lr", type=float, default=0.01, help="learning rate of actor_network")
    parser.add_argument("--critic-lr", type=float, default=0.01, help="learning rate of critic_network")
    parser.add_argument("--gamma", type=float, default=0.95, help="reward discount")
    parser.add_argument("--tau", type=float, default=0.01, help="the rate of soft-update")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--restore", action='store_true', default=False, help='restore or not')
    parser.add_argument("--model-dir", type=str, default='./model', help="directory for saving model")
    parser.add_argument("--max-episode", type=int, default=25000)
    parser.add_argument("--max-step-per-episode", type=int, default=25, help="max step of every episode")
    parser.add_argument("--show", action='store_true', default=False, help='display or not')
    parser.add_argument("--device", default=device, help="torch device")
    parser.add_argument("--load", default=True)

    return parser.parse_args()