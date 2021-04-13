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
import math
from PIL import Image, ImageDraw
 
from moviepy.editor import ImageSequenceClip

def make_movie(policy, env, filename, args):
    with torch.no_grad():
        state = torch.zeros(1, args.hidden_size)
        mask = torch.ones(1,1)
        obss = []
        positions = []
        
        obs = env.reset().astype(np.float32)
        done = False
        while not done:
            obss.append(obs*2)
            result = policy(torch.from_numpy(obs).unsqueeze(0), state, mask)

            action = result['actions']
            state = result['states']
            positions.append(env.get_player_position())

            obs, reward, done, _ = env.step(action.item())
            obs = obs.astype(np.float32)


    observations = [o.transpose(1,2,0) for o in obss]
    clip = ImageSequenceClip(observations, fps=int(30/args.frame_skip))
    clip.write_videofile(filename)

    xmin = min(positions[:][0])
    xmax = max(positions[:][0])
    ymin = min(positions[:][1])
    ymax = max(positions[:][1])

    im = Image.new("RGB", (5000, 5000))
    img1 = ImageDraw.Draw(im)
    lines = []
    for i in range(len(positions)-1):
        lines.append([(1500+int(positions[i][0]), 1500+int(positions[i][1])), (1500+int(positions[i+1][0]), 1500+int(positions[i+1][1]))])
    for line in lines:
        img1.line(line, fill="white", width=3)
    im.save(filename[:-4]+".png")


def evaluate_saved_model():  
    args = parse_a2c_args()
    #TROUVER COMMENT UTILISER LE GPU
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else 
    env = DoomEnvironment(args, is_train=True)
    print(env.num_actions)
    obs_shape = (3, args.screen_height, args.screen_width)

    policy = CNNPolicy(obs_shape, args).to(device)
    args.scenario_dir = "scenarios_transfer_learning/big_combs_test/"
    checkpoint = torch.load("0.pth.tar", map_location=lambda storage, loc: storage)
    policy.load_state_dict(checkpoint['model'])
    policy.eval()
    
    assert args.model_checkpoint, 'No model checkpoint found'
    assert os.path.isfile(args.model_checkpoint), 'The model could not be loaded'
    # This lambda stuff is required otherwise it will try and load on GPU
 
    for i in range(1, 64): #for i in range(args.num_mazes_test):
        env = DoomEnvironment(args, idx=i, is_train=False)
        movie_name = 'videos/3 - 12-04-21 - base/big_combs_test{:0004}.mp4'.format(i)
        print('Creating movie {}'.format(movie_name))
        make_movie(policy, env, movie_name, args)
        
        
if __name__ == '__main__':
    evaluate_saved_model()
    
    
