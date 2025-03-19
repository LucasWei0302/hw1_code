# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import heapq
import math
from collections import deque
from itertools import combinations

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    queue = list()
    queue.append((start, [start]))
    visited = set()
    visited.add(start)

    while len(queue) > 0:
        cpos, cpath = queue.pop(0)
        if maze.isObjective(cpos[0], cpos[1]):
            return cpath
        for nextpos in maze.getNeighbors(cpos[0], cpos[1]):
            if nextpos not in visited:
                nextpath = cpath.copy()
                nextpath.append(nextpos)
                queue.append((nextpos, nextpath))
                visited.add(nextpos)
    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    food = maze.getObjectives()[0]

    priority_queue = []
    heapq.heappush(priority_queue, (0, 0, start, [start]))
    searched = set()

    while priority_queue:
        fvalue, cost, cpos, path = heapq.heappop(priority_queue)

        if cpos == food:
            return path

        if cpos in searched:
            continue
        searched.add(cpos)

        for neighbor in maze.getNeighbors(cpos[0], cpos[1]):
            if neighbor in searched:
                continue
            new_cost = cost + 1
            h = manhattan(neighbor, food)
            heapq.heappush(priority_queue, (new_cost + h, new_cost, neighbor, path + [neighbor]))

    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return astar_multi(maze)

def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])



def mst_cost(points):
    if not points:
        return 0
    edges = [(manhattan(a, b), a, b) for a, b in combinations(points, 2)]
    edges.sort()

    parents = {p: p for p in points}
    def find(p):
        if parents[p] != p:
            parents[p] = find(parents[p])
        return parents[p]
    def union(p1, p2):
        root1, root2 = find(p1), find(p2)
        if root1 != root2:
            parents[root2] = root1

    mst_cost = 0
    edge_count = 0
    for cost, a, b in edges:
        if find(a) != find(b):
            union(a, b)
            mst_cost += cost
            edge_count += 1
            if edge_count == len(points)-1:
                break
    return mst_cost


def heuristic(state):
    cpos, unvisited = state
    if not unvisited:
        return 0
    min_dist = min([manhattan(cpos, food) for food in unvisited])
    return min_dist + mst_cost(unvisited)



def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    food_positions = set(maze.getObjectives())

    priority_queue = []
    heapq.heappush(priority_queue, (0, 0, start, tuple(sorted(food_positions)), [start]))
    searched = set()

    while priority_queue:
        fvalue, cost, cpos, unvisited, path = heapq.heappop(priority_queue)
        if not unvisited:
            return path
        state = (cpos, tuple(sorted(unvisited)))
        if state in searched:
            continue
        searched.add(state)

        for neighbor in maze.getNeighbors(cpos[0], cpos[1]):
            new_food = tuple(sorted(set(unvisited) - {neighbor}))
            new_cost = cost + 1
            h = heuristic((neighbor, new_food))
            heapq.heappush(priority_queue, (h + new_cost, new_cost, neighbor, new_food, path + [neighbor]))




def nearest_food_path(maze, start, food_positions):
    queue = [(start, [start])]
    searched = set()
    while queue:
        cpos, cpath = queue.pop(0)
        if cpos in food_positions:
            return cpath
        if cpos in searched:
            continue
        searched.add(cpos)
        for neighbor in maze.getNeighbors(cpos[0], cpos[1]):
            queue.append((neighbor, cpath + [neighbor]))
    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    food_positions = maze.getObjectives()
    path = []

    while food_positions:
        nearest_path = nearest_food_path(maze, start, food_positions)
        if not nearest_path:
            return path
        path.extend(nearest_path[1:])
        start = nearest_path[-1]
        food_positions.remove(start)
    return path

