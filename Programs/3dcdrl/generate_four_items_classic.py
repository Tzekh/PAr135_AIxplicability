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


def gen_maze(size, cell_width, plot=True, xmin=0, ymin=0, keep_prob=5):
    range_start = 0  # (-size //2)
    range_end = size  # //2

    walls = {}
    for i in range(range_start, range_end):
        for j in range(range_start, range_end):

            if i != range_end - 1 and random.randint(0, 9) < keep_prob:
                # vertical
                start_x = i * cell_width + cell_width + xmin
                end_x = i * cell_width + cell_width + xmin
                start_y = j * cell_width + ymin
                end_y = j * cell_width + cell_width + ymin

                walls[(i, j, i + 1, j)] = (start_x, start_y, end_x, end_y)
                # print('v',(i,j,i+1,j))
            # horizontal
            if j != range_end - 1 and random.randint(0, 9) < keep_prob:
                start_x = i * cell_width + xmin
                end_x = i * cell_width + cell_width + xmin
                start_y = j * cell_width + cell_width + ymin
                end_y = j * cell_width + cell_width + ymin

                walls[(i, j, i, j + 1)] = (start_x, start_y, end_x, end_y)

            # print('h',(i,j,i,j+1))

    extents_x = [range_start * cell_width + xmin,
                 range_start * cell_width + xmin,
                 range_end * cell_width + xmin,
                 range_end * cell_width + xmin,
                 range_start * cell_width + xmin]

    extents_y = [range_start * cell_width + ymin,
                 range_end * cell_width + ymin,
                 range_end * cell_width + ymin,
                 range_start * cell_width + ymin,
                 range_start * cell_width + ymin]
    # if plot:
    #     plt.subplot(1,2,1)
    #     plt.plot(extents_x, extents_y, c='k')

    #     for indx, entry in walls.items():
    #         x0,y0,x1,y1 = entry

    #         #plt.scatter([x0,x1],[y0,y1], c='r')
    #         plt.plot([x0,x1],[y0,y1], c='k')

    # create the neighbours dict
    neighbours = {}
    for i in range(range_start, range_end):
        for j in range(range_start, range_end):
            neighbours[(i, j)] = [(i - 1, j),
                                  (i + 1, j),
                                  (i, j - 1),
                                  (i, j + 1)]

    def valid_neighbour(i, j):
        return i >= range_start and i < range_end and j >= range_start and j < range_end

    def walk(current_i, current_j):
        # print(current_i, current_j)
        visited.add((current_i, current_j))
        n = neighbours[(current_i, current_j)]
        random.shuffle(n)
        for (ni, nj) in n:
            if valid_neighbour(ni, nj) and (ni, nj) not in visited:
                if (current_i, current_j, ni, nj) in walls:
                    # print('wall1:',(current_i, current_j, ni, nj) )
                    # print(walls[(current_i, current_j, ni, nj)])

                    del walls[(current_i, current_j, ni, nj)]
                if (ni, nj, current_i, current_j) in walls:
                    # print('wall2:', (ni, nj, current_i, current_j ) )
                    # print(walls[(ni, nj, current_i, current_j )])
                    del walls[(ni, nj, current_i, current_j)]
                walk(ni, nj)

    visited = set()
    start_i = random.randint(range_start, range_end - 1)
    start_j = random.randint(range_start, range_end - 1)
    walk(start_i, start_j)

    if plot:
        # plt.subplot(1,2,2)
        plt.plot(extents_x, extents_y, c='k')

        for indx, entry in walls.items():
            x0, y0, x1, y1 = entry

            # plt.scatter([x0,x1],[y0,y1], c='r')
            plt.plot([x0, x1], [y0, y1], c='k')

    walls = [w for w in walls.values()]
    exterior = [(x, y) for x, y in zip(extents_x, extents_y)]

    return exterior, walls

def create_maze(base_filepath, filename, size, cell_size):
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

    verticies = []
    wall_cons = []
    wall_idx = 0
    map_point_idx = 10
    output_list = ['// Written by anonymous', 'namespace="zdoom";']

    # create the two map points
    xmin = -size // 2 * cell_size
    ymin = -size // 2 * cell_size

    item_start = 20;
    num_items = 4
    locations = []

    item_tids = [2018, 2019, 2012, 2013]
    colors = ['g', 'r', 'b', 'c']
    details = {}

    items = []

    for i in range(num_items):
        item_i = random.randint(0, size - 1)
        item_j = random.randint(0, size - 1)

        while (item_i, item_j) in locations:
            item_i = random.randint(0, size - 1)
            item_j = random.randint(0, size - 1)

        locations.append((item_i, item_j))

    for loc, tid, idx, col in zip(locations,
                                  item_tids,
                                  range(item_start, item_start + num_items),
                                  colors):
        # print(loc, tid, idx, col)
        item_i, item_j = loc

        item_x = xmin + item_i * cell_size + cell_size / 2
        item_y = ymin + item_j * cell_size + cell_size / 2

        output_list += create_object(item_x, item_y, tid, idx)

        plt.scatter(item_x, item_y, c=col)
        items.append((item_x, item_y, col))

    details['items'] = items

    spawn_i = random.randint(0, size - 1)
    spawn_j = random.randint(0, size - 1)

    while (spawn_i, spawn_j) in locations:
        # print('retry')
        spawn_i = random.randint(0, size - 1)
        spawn_j = random.randint(0, size - 1)

    spawn_x = xmin + spawn_i * cell_size + cell_size / 2
    spawn_y = ymin + spawn_j * cell_size + cell_size / 2

    output_list += create_spawn(spawn_x, spawn_y)
    plt.scatter(spawn_x, spawn_y, c='g', marker='s')
    details['spawn'] = (spawn_x, spawn_y)

    map_point_idx += 1
    
    ##Génération des murs du labyrinthe et de l'extérieur du niveau
    exterior, walls = gen_maze(size, cell_size, xmin=xmin, ymin=ymin, keep_prob=6)
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
    
    #iterate through list to create output text file
    output_string = ''
    for output in output_list:
        output_string += output + '\n'
        
    wad.data['TEXTMAP'].data = output_string.encode()
    wad.to_file(base_filepath +filename) 
    
    
if __name__ == '__main__':
    
    BASE_FILEPATH = "scenarios_transfer_learning/mazes_classic_test/"
    NUM_MAZES = 64
    width=[1]*NUM_MAZES
    rw=[random.randint(5, 7) for i in range(NUM_MAZES)]
    height=[1]*NUM_MAZES
    rh=[random.randint(5, 7) for i in range(NUM_MAZES)]
    
    #Generate NUM_MAZES .was files
    for m in range(0, NUM_MAZES):
        filename = 'custom_scenario_test{:003}.wad'.format(m)
        print('creating maze', filename)

        create_maze(BASE_FILEPATH, filename, size=5, cell_size=160)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
