#Programme pour la génération de labyrinthe. Contient les classe Labyrinth et Tile

import random
from matplotlib import lines as plines
import matplotlib.pyplot as pl

class Labyrinth():

    def __init__(self):
        self.tiles=[]
        self.items=[] #(i, j, tid, idx) with tid showing the type of item and idx it's unique key

        #Data used for generation in Doom
        self.gen_walls={}
        self.gen_exterior=[]

    def Build_tiles(self, width, height, wall_closed=True):
        """
        This function creates every Tile() object in self.tiles
        self.tiles will be a list of lists which last element is indexed self.tiles[width-1][height-1]
        """
        #Initialize some self variables
        self.tiles=[[0]*height for _ in range(width)]
        self.width=width
        self.height=height

        #Create every Tile() object
        for x in range(width):
            for y in range(height):
                self.tiles[x][y]=Tile(self, x, y, wall_closed=wall_closed)


    def Generate(self, start_tile, width=0, height=0, build=False):
        """
        Recurring generation of the labyrinth.
        The input start_tile must be a Tile() and correspond to the first tile visited in the generation algorithm
        If build, it will execute the Build_tiles() function
        """
        #Build or not
        if build:
            self.Build_tiles(width, height)

        #tile is visited
        tile=start_tile
        tile.visited=True

        #Initiate while loop
        neighbours=tile.neighbours(True)
        while neighbours!=[]:
            #choose a random neighbour to visit
            choice=neighbours[random.randint(0,len(neighbours)-1)]

            #unlock the wall we went through in both tile.walls
            dir=(choice.x-tile.x, choice.y-tile.y)
            tile.walls[dir]=False
            choice.walls[(-dir[0], -dir[1])]=False

            #repeat in that new tile
            self.Generate(choice)

            #update the neighbours (unvisited) when going backward
            neighbours=tile.neighbours(True)

    def Generate_rooms(self, ncol, nrow, rw, rh, comb_door = 1, comb_wall=1, central_alley=True, fixed_orientation = False):
        """
        This generates an environment surrounded by walls and containing a grid of rooms such as a chessboard
        The rooms are sized using rowsize and colsize in number of tiles
        In each room, walls are added randomly

        The function returns a usable (exterior, wall) tuple

        Comb parameter:

        The comb parameter must be a float between 0 and 1
        It defines proporiton of comb walls in each room. 0 means no comb walls
        and 1 means just one door in each comb wall.
        """
        width=ncol*rw
        height=nrow*rh
        self.Build_tiles(width, height, wall_closed=False)
        self.punishements=[] #will contain (x,y,idx) tuples for punishement items in combs. (x,y) are coordinates of the tile, not coordinates of items
        self.items=[] #will contain the 4-items (x,y,tid,idx) tuples
        item_tids = [(2018, 20), (2019, 21), (2012, 22), (2013, 23)] #all the tid values for items to gather
        # typical ids_marks to Display() : { 2018:'g' , 2019:'r' , 2012:'b' , 2013:'c' }

        #Close borders and make rooms
        for i in range(width):
            self.tiles[i][0].walls[(0,-1)]=True
            for n in range(1, nrow):
                self.tiles[i][n*rh].walls[(0,-1)]=True
                self.tiles[i][n*rh-1].walls[(0,1)]=True
            self.tiles[i][-1].walls[(0,1)]=True
        for j in range(height):
            self.tiles[0][j].walls[(-1,0)]=True
            for n in range(1, ncol):
                self.tiles[n*rw][j].walls[(-1,0)]=True
                self.tiles[n*rw-1][j].walls[(1,0)]=True
            self.tiles[-1][j].walls[(1,0)]=True

        #Open some doors between each room
        for n in range(nrow):
            for m in range(ncol):
                if n<nrow-1: #si on n'est pas à la pièce la plus haute on ouvre le mur au dessus
                    door=m*rw + random.randint(0,rw-1)
                    self.tiles[door][(n+1)*rh].walls[(0,-1)]=False
                    self.tiles[door][(n+1)*rh-1].walls[(0,1)]=False
                if m<ncol-1: #idem pour mur droit
                    door=n*rh + random.randint(0,rh-1)
                    self.tiles[(m+1)*rw][door].walls[(-1,0)]=False
                    self.tiles[(m+1)*rw-1][door].walls[(1,0)]=False

        #Value that defines the orientation of the comb in each room
        #   0 : horizontal
        #   1 : Vertical ||||
        if not fixed_orientation:
            self.orientations=[[random.randint(0, 1)]*ncol for _ in range(nrow)]
        else :
            self.orientations = [[1] * ncol for _ in range(nrow)]

        #nb_corridor is needed to calculate items placement probabilities
        nb_corridor=0
        for i in range(ncol):
            for j in range(nrow):
                orientation=self.orientations[j][i]
                nb_corridor+=2*rw*orientation+2*rh*(1-orientation)

        #calculus of the corridors where items will be
        corridors=[random.randint(0, nb_corridor-1)]
        rand=random.randint(0,nb_corridor-1)
        for _ in range(3):
            while rand in corridors:
                rand=random.randint(0, nb_corridor-1)
            corridors.append(rand)


        corridor=0 # used to count the corridors and to know where to put an item
        items_left=4 # used to calculate tids of the items
        punishements=0


        #for each room
        for i in range(ncol):
            for j in range(nrow):
                orientation = self.orientations[j][i]

                # x,y are defined so that whatever the orientation, the following remains the same :
                x = rw*orientation+rh*(1-orientation) # is the axe of the main alley
                y = rh*orientation+rw*(1-orientation) # is the axe of the walls of the comb

                if central_alley:
                    # the door value determines the position of the door in each wall
                    # there are problems if door = 0 or y-1
                    door = random.randint(1, y - 2)
                    if i==0 and j==0:
                        self.spawn=(door*(1-orientation), door*orientation)
                else :
                    self.spawn = ((1 - orientation), orientation)

                for k in range(x):

                    #Items placement
                    if corridor in corridors:
                        #Alors on place l'item dans le couloir gauche ou haut (l=0)
                        posx = i*rw + k*orientation+0*(1-orientation)
                        posy = j*rh + 0*orientation+k*(1-orientation)
                        tid, idx = item_tids.pop(random.randint(0, len(item_tids)-1))
                        self.items.append((posx, posy, tid, idx))

                        items_left-=1

                    elif central_alley:
                        #Alors on place les items punitifs
                        punishements+=1
                        posx = i*rw + k*orientation+0*(1-orientation)
                        posy = j*rh + 0*orientation+k*(1-orientation)
                        tid=2012
                        idx=100*punishements + 5 #change the digit here to modify the reward malus when this item is gathered (end corridor item)
                        self.items.append((posx, posy, tid, idx))
                        punishements+=1; idx+=100; self.items.append((posx+0.3*orientation, posy+0.3*(1-orientation), tid, idx)) #2 more punishements to make a barrier
                        punishements+=1; idx+=100; self.items.append((posx-0.3*orientation, posy-0.3*(1-orientation), tid, idx))
                        if door>2: #si il a assez de place on met une punition intermédiaire
                            punishements+=1
                            posx = i*rw + k*orientation+door//2*(1-orientation)
                            posy = j*rh + door//2*orientation+k*(1-orientation)
                            tid=2012
                            idx=100*punishements + 2 #change the digit here to modify the reward malus when this item is gathered (mid corridor item)
                            self.items.append((posx, posy, tid, idx))
                            punishements+=1; idx+=100; self.items.append((posx+0.3*orientation, posy+0.3*(1-orientation), tid, idx))
                            punishements+=1; idx+=100; self.items.append((posx-0.3*orientation, posy-0.3*(1-orientation), tid, idx))

                    corridor+=1

                    #Same thing for the facing corridor
                    if corridor in corridors:
                        #Alors on place l'item dans le couloir bas ou droit (l=y-1])
                        posx = i*rw + k*orientation+(y-1)*(1-orientation)
                        posy = j*rh + (y-1)*orientation+k*(1-orientation)
                        tid, idx = item_tids.pop(random.randint(0, len(item_tids)-1))
                        self.items.append((posx, posy, tid, idx))

                        items_left-=1

                    elif central_alley:
                        #Alors on place les items punitifs
                        punishements+=1
                        posx = i*rw + k*orientation+(y-1)*(1-orientation)
                        posy = j*rh + (y-1)*orientation+k*(1-orientation)
                        tid=2012
                        idx=100*punishements + 5 #change the digit here to modify the reward malus when this item is gathered (end corridor item)
                        self.items.append((posx, posy, tid, idx))
                        punishements+=1; idx+=100; self.items.append((posx+0.3*orientation, posy+0.3*(1-orientation), tid, idx))
                        punishements+=1; idx+=100; self.items.append((posx-0.3*orientation, posy-0.3*(1-orientation), tid, idx))
                        if y-door>2:
                            punishements+=1
                            posx = i*rw + k*orientation+(y+door)//2*(1-orientation)
                            posy = j*rh + (y+door)//2*orientation+k*(1-orientation)
                            tid=2012
                            idx=100*punishements + 2 #change the digit here to modify the reward malus when this item is gathered (mid corridor item)
                            self.items.append((posx, posy, tid, idx))
                            punishements+=1; idx+=100; self.items.append((posx+0.3*orientation, posy+0.3*(1-orientation), tid, idx))
                            punishements+=1; idx+=100; self.items.append((posx-0.3*orientation, posy-0.3*(1-orientation), tid, idx))

                    corridor+=1

                    #On construit les murs, selon la proba comb_wall
                    if random.random()<=comb_wall:
                        for l in range(y):
                            if not central_alley:
                                # the door value determines the position of the door in each wall
                                # there are problems if door = 0 or y-1
                                door = random.randint(1, y - 2)
                            if k<x:
                                posx = i*rw + k*orientation+l*(1-orientation)
                                posy = j*rh + l*orientation+k*(1-orientation)
                                if (random.random()<=comb_door and l!=door):
                                    wall = True
                                else:
                                    wall = False
                                self.tiles[posx][posy].walls[(orientation, 1-orientation)] = wall
                                self.tiles[posx][posy].walls[(-orientation, orientation-1)] = wall





    def Remove_some_walls(self, display=True, ocw=False, p=0):
        """
        This function removes some walls randomly.
        It keeps the border walls closed so that the AI do no escape

        display determines if the labyrinth is displayed at the end of this fonction (using self.Display())
        ocw is an open_closed_wall variable. So if True, when choosing which wall to open, the algorithm necessarily opens a new wall
        p is the probability of opening a second wall after opening one in one tile
        """

        possible_walls=((0,1), (1,0), (0,-1), (-1,0))

        #for each tile
        for _ in self.tiles:
            for tile in _:
                for toto in range(1 + int(random.random() < p)): #Choosing randomly the number of walls to open

                    #if we don't want walls to be 'opened twice'
                    if ocw:
                        possible_walls=[]
                        for wall in tile.walls.keys():
                            if tile.walls[wall]==False: #we keep only walls that are closed
                                possible_walls.append(wall)
                    #if not ocw
                    random.random_wall=possible_walls[random.randint(0,len(possible_walls)-1)]

                    #Keeping borders closed
                    if random.random_wall[0]+tile.x >= 0 and random.random_wall[0]+tile.x < self.width and random.random_wall[1]+tile.y >=0 and random.random_wall[1]+tile.y < self.height:
                        tile.walls[random.random_wall]=False

        #Display or not
        if display:
            self.Display()


    def Display(self, items=False, ids_marks=[]):
        """
        This gives a quick display of the labyrinth using matplotlib
        Using item=True can display items as well.
        It requires to give ids_marks a dictionnary {ids:marks} with marks corresponding to the matplotlib.plot argument
        """
        #Initiate drawing
        line=plines.Line2D
        fig = pl.figure()
        fig.height=self.height+5
        fig.width=self.width+5

        #Initiate wall_plot list and bottom and left border walls
        wall_plot=[0,0]
        wall_plot[0]=line([0, self.width], [0, 0], transform=fig.transFigure, figure=fig)
        wall_plot[1]=line([0, 0], [0, self.height], transform=fig.transFigure, figure=fig)

        #It is enough to build only right and upper wall for each tile...
        #...as the bottom and left walls are the previous tile's upper and right walls
        for _ in self.tiles:
            for tile in _:
                #for each tile we had the two walls to the wall_to_plot list
                if tile.walls[(0,1)]:
                    wall_plot.append(line( [tile.x/self.width, (tile.x+1)/self.width], [(tile.y+1)/self.height, (tile.y+1)/self.height], transform=fig.transFigure, figure=fig))
                if tile.walls[(1,0)]:
                    wall_plot.append(line( [(tile.x+1)/self.width, (tile.x+1)/self.width], [tile.y/self.height, (tile.y+1)/self.height], transform=fig.transFigure, figure=fig))

        #Adding items to the walls as squares
        w=self.width
        h=self.height
        for item in self.items:
            x=item[0]
            y=item[1]
            x=[(x+0.35)/w, (x+0.65)/w, (x+0.65)/w, (x+0.35)/w]
            y=[(y+0.65)/h, (y+0.65)/h, (y+0.35)/h, (y+0.35)/h]
            wall_plot.append( line( [x[0], x[1]], [y[0], y[1]], transform=fig.transFigure, figure=fig, color=ids_marks[item[2]]))
            wall_plot.append( line( [x[1], x[2]], [y[1], y[2]], transform=fig.transFigure, figure=fig, color=ids_marks[item[2]]))
            wall_plot.append( line( [x[2], x[3]], [y[2], y[3]], transform=fig.transFigure, figure=fig, color=ids_marks[item[2]]))
            wall_plot.append( line( [x[3], x[0]], [y[3], y[0]], transform=fig.transFigure, figure=fig, color=ids_marks[item[2]]))
            print('couleur ' + ids_marks[item[2]] + ' en position : ' + str(item[0]) + "," + str(item[1]))

    ##Difficulty : plotting items on pyplot
        #build walls'plot
        fig.lines.extend(wall_plot)

        #And finally plot the figure
        pl.show()

    def Exterior_Wall(self, cell_size):
        """
        Returns the exterior and walls variables needed to generate Doom levels using the programs in 3d_rl GitHub
        wall is a list of tuples representing each wall (x1, y1, x2, y2)
        exterior shapes the physical limits of the Doom world in which the labyrinth will be placed. It needs to be just large enough

        cell_size is needed in pixels
        """
        #this will be the wall list
        gen_walls=[]
        #first make the border walls
        gen_walls.append((0, 0, self.width*cell_size, 0))
        gen_walls.append((self.width*cell_size, 0, self.width*cell_size, self.height*cell_size))
        gen_walls.append((self.width*cell_size, self.height*cell_size, 0, self.height*cell_size))
        gen_walls.append((0, self.height*cell_size, 0, 0))

        #then for each tile, add the bottom and left wall...
        #It is enough to build only left and bottom walls for each tile...
        #...as the upper and right walls are the previous tile's bottom and left walls
        for _ in self.tiles:
            for tile in _:
                #let's also check that we do not build border walls twice
                if tile.walls[(0,-1)] and tile.y>0:
                    gen_walls.append((tile.x*cell_size, tile.y*cell_size, tile.x*cell_size + cell_size, tile.y*cell_size))
                if tile.walls[(-1,0)] and tile.x>0:
                    gen_walls.append((tile.x*cell_size, tile.y*cell_size, tile.x*cell_size, tile.y*cell_size + cell_size))

        #exterior is just the list representing the polygon of the world limits
        exterior=[(-self.width*cell_size,-self.height*cell_size), (self.width*cell_size*2, -self.height*cell_size), (self.width*cell_size*2, self.height*cell_size*2), (-self.width*cell_size, self.height*cell_size*2), (-self.width*cell_size,-self.height*cell_size)]

        return exterior, gen_walls

    def get_items_spawn(self):
        return self.items, self.spawn

class Tile():

    def __init__(self, laby, x, y, wall_closed=True):
        self.labyrinth=laby
        self.x=x
        self.y=y
        self.walls={(0,1):wall_closed, (1,0):wall_closed, (0,-1):wall_closed, (-1,0):wall_closed} #walls are indexed by their direction from the tile's point of vue
        self.visited=False
        self.value=0 #value may indicate if the tile contains an item or the player's span or anything worth knowing

    def neighbours(self, unvisited_neighbours_only=False):
        #return a list of list containing the coordinates of the tile's neighbours within the Labyrinth borders
        #unvisited_neighbours_only allows to return unvisited neighbours only
        neighbours=[]
        dirs=[(1,0), (-1,0), (0,1), (0,-1)]
        for dir in dirs:
            if self.x+dir[0]>=0 and self.x+dir[0]<self.labyrinth.width and self.y+dir[1]<self.labyrinth.height and self.y+dir[1]>=0:
                if not unvisited_neighbours_only:
                    neighbours.append(self.labyrinth.tiles[self.x+dir[0]][self.y+dir[1]])
                else:
                    if self.labyrinth.tiles[self.x+dir[0]][self.y+dir[1]].visited==False:
                        neighbours.append(self.labyrinth.tiles[self.x+dir[0]][self.y+dir[1]])

        return neighbours


def main():
    mylaby=Labyrinth()
    mylaby.Generate_rooms(1,1,10,8,1,1)
    mylaby.Display(items=True, ids_marks={ 2018:'g' , 2019:'r' , 2012:'b' , 2013:'c' })

