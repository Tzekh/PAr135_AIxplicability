
import time, logging

import torch

import torch.utils.tensorboard


# !/usr/bin/env python3
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



def evaluate_saved_model():
    args = parse_a2c_args()
    args2 = parse_a2c_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_updates = int(args.num_frames) // args.num_steps // args.num_environments

    # Writer will output to ./runs/ directory by default
    writer = torch.utils.tensorboard.SummaryWriter()


    train_envs = MultiEnv(args.simulator, args.num_environments, args, is_train=True)

    # Création des environnements de test des niveaux classiques
    args2.scenario_dir = "scenarios_transfer_learning/mazes_classic_test/"
    args2.scenario = "custom_scenario_test{:003}.cfg"
    classic_test_envs = MultiEnv(args.simulator, args.num_environments, args2, is_train=False)
    # Création des environnements de test des niveaux peignes
    args2.scenario_dir = "scenarios_transfer_learning/little_combs_test/"
    little_combs_test_envs = MultiEnv(args.simulator, args.num_environments, args2, is_train=False)
    args2.scenario_dir = "scenarios_transfer_learning/medium_combs_test/"
    medium_combs_test_envs = MultiEnv(args.simulator, args.num_environments, args2, is_train=False)


    obs_shape = train_envs.obs_shape

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

    obs = little_combs_test_envs.reset()

    num_checkpoints = 355

    for j in range(num_checkpoints):
        if j%8 == 0:
            checkpoint_filename = '/home/adam/Bureau/Transfer Learning/FINAL/checkpoint_{}.pth.tar'.format(str(j+1))
            agent.load_model(checkpoint_filename)

            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            mean_rewards_classic, game_times_classic = agent.evaluate(classic_test_envs, j, total_num_steps)
            mean_rewards_little, game_times_little = agent.evaluate(little_combs_test_envs, j, total_num_steps)
            mean_rewards_medium, game_times_medium = agent.evaluate(medium_combs_test_envs, j, total_num_steps)

            writer.add_scalar("Reward classic levels", mean_rewards_classic, (j+1)*100)
            writer.add_scalar("Reward little combs levels", mean_rewards_little, (j+1)*100)
            writer.add_scalar("Reward medium combs levels", mean_rewards_medium, (j+1)*100)
            print(j)



if __name__ == '__main__':
    evaluate_saved_model()


