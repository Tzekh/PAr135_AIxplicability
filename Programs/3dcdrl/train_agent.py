#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:54:38 2019

@author: edward
"""
import time, logging

import torch

from arguments import parse_a2c_args
from multi_env import MultiEnv
from models import CNNPolicy
from a2c_agent import A2CAgent
from utils import initialize_logging
import torch.utils.tensorboard
from doom_environment import DoomEnvironment


def train():
    args = parse_a2c_args()
    args2 = parse_a2c_args()
    output_dir = initialize_logging(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_updates = int(args.num_frames) // args.num_steps // args.num_environments
    # Create the train and test environments with Multiple processes
    train_envs = MultiEnv(args.simulator, args.num_environments, args, is_train=True)

    #Création des environnements de test des niveaux classiques
    args2.scenario_dir = "scenarios_transfer_learning/mazes_classic_test/"
    args2.scenario = "custom_scenario_test{:003}.cfg"
    classic_test_envs = MultiEnv(args.simulator, args.num_environments, args2, is_train=False)
    #Création des environnements de test des niveaux peignes
    args2.scenario_dir = "scenarios_transfer_learning/little_combs_test/"
    little_combs_test_envs = MultiEnv(args.simulator, args.num_environments, args2, is_train=False)
    args2.scenario_dir = "scenarios_transfer_learning/medium_combs_test/"
    medium_combs_test_envs = MultiEnv(args.simulator, args.num_environments, args2, is_train=False)


    test_envs = MultiEnv(args.simulator, args.num_environments, args, is_train=False)

    # Writer will output to ./runs/ directory by default
    writer = torch.utils.tensorboard.SummaryWriter()
    
    obs_shape = train_envs.obs_shape
    
    # The agent's policy network and training algorithm A2C
    policy = CNNPolicy(obs_shape, args).to(device)
    agent = A2CAgent(policy, 
                     args.hidden_size,
                     value_weight=args.value_loss_coef, 
                     entropy_weight=args.entropy_coef, 
                     num_steps=args.num_steps, 
                     num_parallel=args.num_environments,
                     gamma=args.gamma,
                     lr=args.learning_rate,
                     opt_alpha=args.alpha,
                     opt_momentum=args.momentum,
                     max_grad_norm=args.max_grad_norm)
    
    start_j = 0
    if args.reload_model:
        checkpoint_idx = args.reload_model.split(',')[1]
        checkpoint_filename = '{}models/base_line.pth.tar'.format(output_dir)
        agent.load_model(checkpoint_filename)
        start_j = 0 #(int(checkpoint_idx) // args.num_steps // args.num_environments) + 1
        
    obs = train_envs.reset()
    start = time.time()
    nb_of_saves = 0
    
    for j in range(start_j, num_updates):
        print("------", j/num_updates*100 , "-------")


        # Test des performances du modèle
        if not args.skip_eval and j % args.eval_freq == 0:
            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            mean_rewards_classic, game_times_classic = agent.evaluate(classic_test_envs, j, total_num_steps)
            mean_rewards_little, game_times_little = agent.evaluate(little_combs_test_envs, j, total_num_steps)
            mean_rewards_medium, game_times_medium = agent.evaluate(medium_combs_test_envs, j, total_num_steps)

           # succes_classic = sum([1 if i!=525 else 0 for i in game_times_classic])/16
          #  succes_little = sum([1 if i!=525 else 0 for i in game_times_little])/16
           # succes_medium = sum([1 if i!=525 else 0 for i in game_times_medium])/16

            writer.add_scalar("Reward classic levels", mean_rewards_classic, j)
            writer.add_scalar("Reward little combs levels", mean_rewards_little, j)
            writer.add_scalar("Reward medium combs levels", mean_rewards_medium, j)
           # writer.add_scalar("Success rate classic levels", succes_classic, j)
           # writer.add_scalar("Success rate little combs levels", succes_little, j)
           # writer.add_scalar("Success rate medium combs levels", succes_medium, j)


        for step in range(args.num_steps):
            action = agent.get_action(obs, step)
            obs, reward, done, info = train_envs.step(action)
            agent.add_rewards_masks(reward, done, step)



        report = agent.update(obs)
        
        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            save_num_steps = (start_j) * args.num_environments * args.num_steps
            FPS = int((total_num_steps - save_num_steps) / (end - start)),
            
            logging.info(report.format(j, total_num_steps, FPS))  
        
        if j % args.model_save_rate == 0:
            nb_of_saves +=1
            agent.save_policy2(nb_of_saves, args, output_dir)


        
    # cancel the env processes    
    train_envs.cancel()
    test_envs.cancel()
    
    
if __name__ == '__main__':
    train()
    
       