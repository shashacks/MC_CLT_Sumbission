
import spinup
from spinup.user_config import DEFAULT_BACKEND
from spinup.utils.run_utils import ExperimentGrid
from spinup.utils.serialization_utils import convert_json
from spinup.utils.test_policy import load_policy_and_env, run_policy
import argparse
import gym
import json
import os, subprocess, sys
import os.path as osp
import string
import torch
from copy import deepcopy
from textwrap import dedent
import sys
import numpy as np
import time
import argparse

# python -m spinup.test --alg ppg --env Hopper-v2 --render True --path data/Hopper/Hopper_s5000/pyt_save/model.pt

def eval(args):
    alg = args.alg
    env_name = args.env
    env = gym.make(env_name)
    render = args.render
    path = os.path.join(os.getcwd(), args.path)

    seed = 10000 + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    env.seed(seed)

    max_ep_len=1000
    
    pi = torch.load(path)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    n = 0

    vals = []

    while n < 100:
        if render:
            env.render()
            time.sleep(1e-3)

        a = pi.act(torch.as_tensor(o, dtype=torch.float32))
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            vals.append(ep_ret)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    print(alg + ': ' + env_name)
    print('The number of episodes: {}'.format(len(vals)))
    print('mean: {}'.format(np.mean(vals)))
    print('std: {}'.format(np.std(vals)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--alg', type=str, default='ppg')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    eval(args)
    

if __name__ == '__main__':
    main()
    