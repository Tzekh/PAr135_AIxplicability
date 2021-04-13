#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:19:33 2018

@author: anonymous
"""
import os
import torch
import numpy as np
from arguments import parse_a2c_args
from multi_env import MultiEnv
from models import CNNPolicy
from a2c_agent import A2CAgent
from utils import initialize_logging
from doom_environment import DoomEnvironment
 
from moviepy.editor import ImageSequenceClip

def get_results(policy, env, args):
    with torch.no_grad():
        state = torch.zeros(1, args.hidden_size)
        mask = torch.ones(1,1)

        obs = env.reset().astype(np.float32)

        episode_duration = 0
        total_reward =0

        done = False
        while not done:
            result = policy(torch.from_numpy(obs).unsqueeze(0), state, mask)

            action = result['actions']
            state = result['states']

            episode_duration +=1
            obs, reward, done, _ = env.step(action.item())
            total_reward += reward
            obs = obs.astype(np.float32)

    return [total_reward, episode_duration]

def evaluate_saved_model():

    args = parse_a2c_args()
    # TROUVER COMMENT UTILISER LE GPU
    device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else
    env = DoomEnvironment(args, is_train=False)
    print(env.num_actions)
    obs_shape = (3, args.screen_height, args.screen_width)

    policy = CNNPolicy(obs_shape, args).to(device)
    results = []

    for model in range(0,2):
        checkpoint = torch.load(str(model)+".pth.tar", map_location=lambda storage, loc: storage)
        policy.load_state_dict(checkpoint['model'])
        policy.eval()

        assert args.model_checkpoint, 'No model checkpoint found'
        assert os.path.isfile(args.model_checkpoint), 'The model could not be loaded'
        # This lambda stuff is required otherwise it will try and load on GPU

        results.append([])

        for i in range(args.num_mazes_test):
            env = DoomEnvironment(args, idx=i, is_train=False)
            results[model].append(get_results(policy, env, args))
            print(i)

        success_rate = 0
        average_reward = 0
        average_time = 0

        for res in results[model]:
            if  res[1] < 525 :
                success_rate += 1
                average_time += res[1]
            average_reward += res[0]

        if success_rate!=0:
            average_time /= success_rate
        success_rate /= args.num_mazes_test
        average_reward /= args.num_mazes_test

        print(success_rate, average_reward, average_time)

    time_diff = 0
    finished_levels = 0

    for i in range(args.num_mazes_test):
        if results[1][i][1] < 525:
            finished_levels += 1
            time_diff += results[1][i][1] - results[0][i][1]

    print(time_diff/finished_levels)
        
        
if __name__ == '__main__':
    evaluate_saved_model()
    
    
