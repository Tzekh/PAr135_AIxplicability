#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:35:10 2018

@author: anonymous

cd /home/adam/Bureau
"""

import shutil
import random
import numpy as np

#These are used to write the wad file
from omg.wad import WAD
from scenario_generation.maze_functions import create_green_armor, create_red_armor, create_line_def, create_map_point, create_vertex, create_object
from scenario_generation.maze_functions import create_sector, create_side_def, create_spawn, gen_random_maze, create_red_pillar, create_green_pillar
import level_maker

import json


    
def create_maze(base_filepath, filename, width, height, rw, rh, cell_size):
    # load the base files
    BASE_WAD = 'custom_scenario.wad'
    wad = WAD('scenarios/basefiles/' + BASE_WAD)
    BASE_CFG = 'custom_scenario.cfg'
    cfg_filename = '{}{}.cfg'.format(base_filepath,filename[:-4])
    shutil.copy('scenarios/basefiles/' + BASE_CFG, cfg_filename)
    
    #dealing with filename errors
    if '/' in filename:
        wad_filename = filename.split('/')[-1]
    else:
        wad_filename = filename
    
    # change the maze name in .cfg file
    # Read in the file
    with open('scenarios/basefiles/' + BASE_CFG, 'r') as file:
      filedata = file.read()
    
    # Replace the target string
    filedata = filedata.replace(BASE_WAD, wad_filename)
    
    # Write the file out again
    with open(cfg_filename, 'w') as file:
      file.write(filedata)    
    
    #Initializing some variables
    details = {}
    verticies = []
    wall_cons = []
    wall_idx = 0
    map_point_idx = 10
    output_list = ['// Written by anonymous', 'namespace="zdoom";']

    # create the two map points
    xmin = -0
    ymin = 0
    
    
    
    
    map_point_idx += 1

    #Génération des murs et de l'extérieur

    ext_height = 1600
    ext_width = 1600
    padding = 0
    exterior = [(-padding, -padding) , (-padding, ext_height+padding), (ext_width+padding, ext_height+padding), (ext_width+padding, -padding), (-padding, -padding) ]
    walls, spawn = level_maker.draw_level(height=ext_height, width=ext_width)


    verticies += exterior[:-1]
    details['exterior'] = exterior[:-1]
    details['walls'] = walls
    
    #???
    with open(base_filepath+filename[:-4]+'.json', 'w') as f:
        json.dump(details, f)    
    
    #???
    for k in range(4):
        wall_cons.append((wall_idx + k, wall_idx + ((k +1)%4)))    
    wall_idx += 4    
    
    #Conversion des murs en verticies
    pad = 8 #épaisseur des murs
    for wall in walls:
        x0,y0,x1,y1 = wall
        
        # On regarde si le mur et vertical ou non
        # Ajout d'une épaisseur aux murs
        if x0 == x1:
            verticies += [(x0-pad, y0), (x1+pad, y0),
                      (x1+pad, y1), (x0-pad, y1)]
        else:
            verticies += [(x0, y0-pad), (x1, y0-pad),
                      (x1, y1+pad), (x0, y1+pad)]           
        
        for k in range(4):
            wall_cons.append((wall_idx + k, wall_idx + ((k +1)%4)))
        wall_idx += 4          

    # Création des vertex, des line def et des side def (possibilité de modifier les textures)
    for vx, vy in verticies:
        output_list += create_vertex(vx, vy)
    
    for id1, id2 in wall_cons:
        output_list += create_line_def(id1,id2)
        output_list += create_side_def() 
    
    output_list += create_sector()
    
    ##Placement des items et du spawn
    spawn = (spawn[0], spawn[1])

    #output_list += create_object(xmin + spawn[0]*cell_size + cell_size*1.5, ymin + spawn[1]*cell_size + cell_size/2, 2018, 50)
    output_list += create_spawn(spawn[0], spawn[1])
    details['spawn'] = spawn
    
    #iterate through list to create output text file
    output_string = ''
    for output in output_list:
        output_string += output + '\n'
        
    wad.data['TEXTMAP'].data = output_string.encode()
    wad.to_file(base_filepath +filename) 
    
    
if __name__ == '__main__':
    
    BASE_FILEPATH = "scenarios_transfer_learning/scenes/"
    NUM_MAZES = 1
    width=[1]*NUM_MAZES
    rw=[random.randint(5, 8) for i in range(NUM_MAZES)]
    height=[1]*NUM_MAZES
    rh=[random.randint(1, 1) for i in range(NUM_MAZES)]
    
    #Generate NUM_MAZES .was files
    for m in range(0, NUM_MAZES):
        filename = 'custom_scenario{:003}.wad'.format(m)
        print('creating maze', filename)
    
        create_maze(BASE_FILEPATH, filename, width[m], height[m], rw[m], rh[m], 200)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
