# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:29:10 2022

@author: danie
"""

import numpy as np
field = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1,  0,  0,  1,  0,  1,  1,  0,  1,  1,  0,  0,  1,  0,  0,  0, -1],
                [-1,  0, -1,  1, -1,  0, -1,  1, -1,  0, -1,  1, -1,  1, -1,  0, -1],
                [-1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,  1,  1,  1,  1,  1, -1],
                [-1,  1, -1,  1, -1,  0, -1,  1, -1,  0, -1,  1, -1,  0, -1,  1, -1],
                [-1,  0,  1,  0,  1,  1,  1,  1,  0,  1,  1,  1,  0,  1,  0,  1, -1],
                [-1,  1, -1,  1, -1,  0, -1,  1, -1,  1, -1,  1, -1,  0, -1,  1, -1],
                [-1,  1,  1,  1,  1,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  1, -1],
                [-1,  0, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                [-1,  0,  0,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  0, -1],
                [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  0, -1],
                [-1,  1,  1,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  1, -1],
                [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  0, -1,  0, -1,  1, -1],
                [-1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1, -1],
                [-1,  0, -1,  1, -1,  1, -1,  0, -1,  1, -1,  1, -1,  1, -1,  0, -1],
                [-1,  0,  0,  1,  0,  1,  0,  1,  0,  0,  1,  1,  1,  0,  0,  0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
n=3
ownpos = [1,2]
expand_f = np.ones(tuple(s+2*n for s in field.shape)) *-1
expand_f[tuple(slice(n,-n) for s in field.shape)] = field
f_ac = expand_f[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]

expmap = np.zeros(f_ac.shape)
for i in range(4):
    obs = 0
    t= 0
    while t<4 and obs !=-1:
        if f_ac[4-t,4] == -1:
            obs ==-1
            break
        elif expmap[4-t,4] != 0:
            pass
        else:
            expmap[4-t,4] = -1
        t+=1
    expmap = np.rot90(expmap)
    f_ac = np.rot90(f_ac)

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


bombs = [[(1,5),4],[(2,1),1]]
tempexpmap = np.ones((6,17,17))
for bomb in bombs:
    tempexpmap.itemset((bomb[1],bomb[0][0],bomb[0][1]),-1)
    tempexpmap.itemset((bomb[1]+1,bomb[0][0],bomb[0][1]),-1)


for ind,tmap in enumerate(tempexpmap):
    explobombs = np.array(np.where(tmap == -1)).T
    for x,y in explobombs:
        for i in range(4):
            if field[x,y+i] !=-1:  
                tempexpmap.itemset((ind,x,y+i),0)
            else:
                break
        for i in range(4):
            if field[x,y-i] !=-1:
                tempexpmap.itemset((ind,x,y-i),0)
            else:
                break
        for i in range(4):
            if field[x+i,y] !=-1:
                tempexpmap.itemset((ind,x+i,y),0)
            else:
                break
        for i in range(4):
            if field[x-i,y] !=-1:
                tempexpmap.itemset((ind,x-i,y),0)
            else:
                break



matrix = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
,[0,0,1,1,0,1,0,0,1,1,0,0,0,0,1,1,0]
,[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0]
,[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
,[0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0]
,[0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0]
,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0]
,[0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0]
,[0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0]
,[0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,1,0]
,[0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0]
,[0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0]
,[0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0]
,[0,1,0,0,1,0,1,1,0,0,0,0,1,0,0,1,0]
,[0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0]
,[0,1,1,0,0,0,0,0,0,1,1,0,0,0,1,1,0]
,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


def safespace(ownpos,field,tempexpmap):
    selfx,selfy = ownpos
    field = np.tile(np.abs(np.abs(field)-1).T,(6,1,1))
    tempmovemap = np.min(np.stack((tempexpmap,field),axis=-1),axis=-1).astype(int)
    ends = [list(x) for x in np.array(np.where(tempmovemap[-1] == 1)).T if np.linalg.norm(x-ownpos) <6]
    safespacedir = []
    pospaths = []
    grid0 = Grid(matrix=tempmovemap[1].T)
    start0 = grid0.node(selfx,selfy)
    finder = AStarFinder()

    end0 = [list(x) for x in np.array(np.where(tempmovemap[1] == 1)).T if np.linalg.norm(x-[selfx,selfy]) <2.1]
    pospath0 = []
    for end in end0:
        endnode = grid0.node(end[0],end[1])
        path, runs = finder.find_path(start0, endnode, grid0)
        if path != []:
            pospath0.append(path[-1])
            
            
    print(pospath0)
    for path1 in pospath0:
        grid1 = Grid(matrix=tempmovemap[2].T)
        start1 = grid1.node(path1[0],path1[1])
        finder = AStarFinder()
        end1 = [list(x) for x in np.array(np.where(tempmovemap[2] == 1)).T if np.linalg.norm(x-[path1[0],path1[1]]) <2.1]
        pospath1 = []
        for end in end1:
            endnode = grid0.node(end[0],end[1])
            path, runs = finder.find_path(start1, endnode, grid1)
            if path != []:
                pospath1.append(path[-1])
        print(pospath1)
        for path2 in pospath1:
            grid2 = Grid(matrix=tempmovemap[3].T)
            start2 = grid2.node(path2[0],path2[1])
            finder = AStarFinder()
            end2 = [list(x) for x in np.array(np.where(tempmovemap[3] == 1)).T if np.linalg.norm(x-[path2[0],path2[1]]) <2.1]
            pospath2 = []
            for end in end2:
                endnode = grid2.node(end[0],end[1])
                path, runs = finder.find_path(start2, endnode, grid2)
                if path != []:
                    pospath2.append(path[-1])
            print(pospath2)

    
    
    
    
# =============================================================================
# for finendn in ends:
# 
#     currpath = []
#     x = np.copy(selfx)
#     y = np.copy(selfy)
#     for temp in tempmovemap:
#         tempend = []
#         for end in tempend: 
#             grid = Grid(matrix=temp)
#             start = grid.node(x,y)
#             finder = AStarFinder()
#             end = grid.node()
#             path, runs = finder.find_path(start, end, grid)
#             if len(path)> 1 and len(path)<3:
#                 currpath.append(path)
# =============================================================================
safespace(ownpos,field,tempexpmap)











def astar(f_ac,expmap):
    pathtodir = {
        (4,3):0,
        (5,4):1,
        (4,5):2,
        (3,4):3
        }
    f_ac_mod = np.abs(np.abs(f_ac)-1)
    ends = list(set(zip(*np.array(np.where(expmap == 0)).tolist())) & set(zip(*np.array(np.where((f_ac == 0))).tolist())))
    safespacedir = []
    pospath = []
    for endn in ends:
        grid = Grid(matrix= f_ac_mod)
        start = grid.node(4, 4)
        finder = AStarFinder()
        end = grid.node(endn[1],endn[0])
        path, runs = finder.find_path(start, end, grid)
        if len(path)> 1 and len(path)<6:
            pospath.append(path)
    if pospath != []:
        for ppath in pospath:
            safespacedir.append(pathtodir.get(ppath[1]))
        return False,safespacedir
    return True,None
#print(astar(f_ac,expmap))






ends = np.array(np.where(f_ac == 0)).T
for (i_e, j_e) in ends:
    pass

# =============================================================================
# 
# bombdes = np.copy(f_ac)
# # feature 4
# for i in range(4):
#     if bombdes[3,2] == -1:
#         bombdes[0:3,:] =0 
#     print(bombdes)
#     bombdes=np.rot90(bombdes)
# print(np.array(np.where(f_ac == 0)
# =============================================================================
    
