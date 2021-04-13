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

def evaluate_saved_model(models, models_dir):

    args = parse_a2c_args()
    # TROUVER COMMENT UTILISER LE GPU
    device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else

    #création des environnements

    #création des environnements little-combs

    little_comb_env = []
    for i in range(50):
        args.scenario_dir = "scenarios_transfer_learning/little_combs_test/"
        args.scenario = "custom_scenario_test{:003}.cfg"
        little_comb_env.append(DoomEnvironment(args,idx=i, is_train=False))

    #création des environnements big-combs

    big_comb_env = []
    for i in range(50):
        args.scenario_dir = "scenarios_transfer_learning/big_combs_test/"
        args.scenario = "custom_scenario_test{:003}.cfg"
        big_comb_env.append(DoomEnvironment(args, idx=i,is_train=False))

    #création des environnements classic

    classic_env = []
    for i in range(50):
        args.scenario_dir = "scenarios_transfer_learning/mazes_classic_test/"
        args.scenario = "custom_scenario_test{:003}.cfg"
        classic_env.append(DoomEnvironment(args, idx=i,is_train=False))

    #création des environnements other levels


    medium_comb_env = []
    for i in range(16):
        args.scenario_dir = "scenarios_transfer_learning/medium_combs_test/"
        args.scenario = "custom_scenario_test{:003}.cfg"
        medium_comb_env.append(DoomEnvironment(args, idx=i,is_train=False))


    obs_shape = (3, args.screen_height, args.screen_width)

    policy = CNNPolicy(obs_shape, args).to(device)

    resultat = []

    for model in models:
        checkpoint = torch.load(models_dir+"/checkpoint_"+str(model)+".pth.tar", map_location=lambda storage, loc: storage)
        policy.load_state_dict(checkpoint['model'])
        policy.eval()

        resultat.append(model)

        assert args.model_checkpoint, 'No model checkpoint found'
        assert os.path.isfile(args.model_checkpoint), 'The model could not be loaded'
        # This lambda stuff is required otherwise it will try and load on GPU

        # evaluation sur les niveaux classiques

        results = []

        for i in range(50):
            env = classic_env[i]
            results.append(get_results(policy, env, args))

        print("Classic levels evaluation is done")

        success_rate = 0
        average_reward = 0

        for res in results:
            if res[1] < 525:
                success_rate += 1
            average_reward += res[0]

        success_rate /= args.num_mazes_test
        average_reward /= args.num_mazes_test

        resultat.append([success_rate, average_reward])

        # evaluation sur les little combs

        results = []

        for i in range(50):
            env = little_comb_env[i]
            results.append(get_results(policy, env, args))

        print("Little combs evaluation is done")

        success_rate = 0
        average_reward = 0

        for res in results:
            if res[1] < 525:
                success_rate += 1
            average_reward += res[0]

        success_rate /= args.num_mazes_test
        average_reward /= args.num_mazes_test

        resultat.append([success_rate, average_reward])


        # evaluation sur les big combs

        results = []

        for i in range(50):
            env = big_comb_env[i]
            results.append(get_results(policy, env, args))
        print("Big combs evaluation is done")

        success_rate = 0
        average_reward = 0

        for res in results:
            if res[1] < 525:
                success_rate += 1
            average_reward += res[0]

        success_rate /= args.num_mazes_test
        average_reward /= args.num_mazes_test

        resultat.append([success_rate, average_reward])


        # evaluation sur les autres niveaux

        results = []

        for i in range(16):
            env = medium_comb_env[i]
            results.append(get_results(policy, env, args))
        print("Other levels evaluation is done")

        success_rate = 0
        average_reward = 0

        for res in results:
            if res[1] < 525:
                success_rate += 1
            average_reward += res[0]

        success_rate /= args.num_mazes_test
        average_reward /= args.num_mazes_test

        resultat.append([success_rate, average_reward])

        print("Checkpoint "+str(model)+" has been evaluated")

    print(resultat)

        
        
if __name__ == '__main__':

    models = [i*8+1 for i in range(44)]
    evaluate_saved_model(models)
    
    
