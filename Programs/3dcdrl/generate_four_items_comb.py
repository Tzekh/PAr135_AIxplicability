#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:35:10 2018

@author: anonymous

cd /home/adam/Bureau
"""

import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from new_gen_maze import Labyrinth, Tile

#These are used to write the wad file
from omg.wad import WAD
from scenario_generation.maze_functions import create_green_armor, create_red_armor, create_line_def, create_map_point, create_vertex, create_object
from scenario_generation.maze_functions import create_sector, create_side_def, create_spawn, gen_random_maze, create_red_pillar, create_green_pillar

import json

def gen_maze(width, height, rw, rh, cell_size):
    #This function uses the Labyrinth class to return the exterior and wall variables used in the following functions
    my_laby=Labyrinth()
    my_laby.Generate_rooms(width, height, rw,rh, 1, 1)
    return my_laby.Exterior_Wall(cell_size)

def place_spawn_items(width, height, cell_size, xmin, ymin, punishement=True):
    #Still the original function
    """
    Returns the list of the items coordinates tuples, output needed to create_items and spawn location
    """
    #Initialize
    item_start = 20;
    num_items = 4
    output=[]
    items = []
    locations=[]
    punishements=[]
    """
    #Place spawn at random
    spawn_i = random.randint(0, width-1)
    spawn_j = random.randint(0, height-1) 
    locations.append((spawn_i,spawn_j))"""
    #Place spawn not randomly
    spawn_i = 0
    spawn_j = height//2
        
    #Generate output
    spawn = (xmin + spawn_i*cell_size + cell_size/2, ymin + spawn_j*cell_size + cell_size/2)
    
    #this chooses which items 
    item_tids = [2018, 2019, 2012, 2013]
    colors = ['g','r', 'b', 'c']

    #let's place the items
    for i in range(num_items):
        item_i = random.randint(0, width-1)
        #item_j = random.randint(0, height-1)
        item_j = random.randint(0, 1)*(height-1)
        
        #This avoid items superposition
        while (item_i, item_j) in locations:
            item_i = random.randint(0, width-1)
            item_j = random.randint(0, 1)*(height-1)  
        locations.append((item_i, item_j))
             
        #add in the wad output
        item_x = xmin + item_i*cell_size + cell_size/2
        item_y = ymin + item_j*cell_size + cell_size/2
        output += create_object(item_x, item_y, item_tids[i], item_start + i)
        
        #add the item
        items.append((item_x, item_y, colors[i]))
    
    #let's place the punishement items
    for i in range(width):
        #place in upper combs
        for j in range(height//2):
            if not (0,j) in locations:
                idx=100*len(punishements)+2-j%((height//2)/2)
                item_x = xmin + i*cell_size + cell_size/2
                item_y = ymin + j*cell_size + cell_size/2
                tid=2013
                output += create_object(item_x, item_y, tid, idx) # unvisible=True)
                items.append((item_x, item_y, colors[2]))
        #place in lower combs
        for j in range(h//2+1,height):
            if not (width, j) in locations:
                idx=100*len(punishements)+(j-height//2)%((height-height//2)/2)
                item_x = xmin + i*cell_size + cell_size/2
                item_y = ymin + j*cell_size + cell_size/2
                tid=2013
                output += create_object(item_x, item_y, tid, idx) # unvisible=True)
                items.append((item_x, item_y, colors[2]))
    
    return items, output, spawn
        
    
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
    
    ##Génération des murs du labyrinthe et de l'extérieur du niveau
    my_laby=Labyrinth()
    my_laby.Generate_rooms(1,1, rw, rh, central_alley=True, fixed_orientation=True)
    exterior, walls = my_laby.Exterior_Wall(cell_size)
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
    items, spawn = my_laby.get_items_spawn()
    details['items'] = items
    for item in items:
        x=xmin + item[0]*cell_size + cell_size/2
        y=ymin + item[1]*cell_size + cell_size/2
        if item[3] > 24 :
            output_list += create_object(x, y, item[2], item[3], invisible= True)
        else :
            output_list += create_object(x, y, item[2], item[3], invisible= False)
    #output_list += create_object(xmin + spawn[0]*cell_size + cell_size*1.5, ymin + spawn[1]*cell_size + cell_size/2, 2018, 50)
    output_list += create_spawn(xmin + spawn[0]*cell_size + cell_size/2, ymin + spawn[1]*cell_size + cell_size/2)
    details['spawn'] = spawn
    
    #iterate through list to create output text file
    output_string = ''
    for output in output_list:
        output_string += output + '\n'
        
    wad.data['TEXTMAP'].data = output_string.encode()
    wad.to_file(base_filepath +filename) 
    
    
if __name__ == '__main__':
    
    BASE_FILEPATH = "scenarios_transfer_learning/little_combs_test/"
    NUM_MAZES = 256
    width=[1]*NUM_MAZES
    rw=[random.randint(5, 7) for i in range(NUM_MAZES)]
    height=[1]*NUM_MAZES
    rh=[random.randint(5, 7) for i in range(NUM_MAZES)]
    
    #Generate NUM_MAZES .was files
    for m in range(0, NUM_MAZES):
        filename = 'custom_scenario_test{:003}.wad'.format(m)
        print('creating maze', filename)
    
        create_maze(BASE_FILEPATH, filename, width[m], height[m], rw[m], rh[m], 200)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
