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
from collections import deque

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)
from collections import deque

def bfs(maze):
    """
    Runs BFS to find the shortest path from the start position to each dot.

    @param maze: The maze to execute the search on.

    @return paths: A dictionary where keys are goal positions and values are the shortest paths to those goals.
    """
    start = maze.getStart()  
    goals = set(maze.getObjectives())  
    queue = deque([(start, [start])]) 
    visited = set()  
    parent = {}
    path = {}

    # BFS loop
    while len(queue) > 0:
        current, path = queue.popleft()  
        
        if current in goals: 
            return path

        if current in visited:
            continue
        visited.add(current)

        # Explore neighbors
        for neighbor in maze.getNeighbors(*current):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor])) 
    return []  

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    path = astar_multi(maze)
    return path

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    path = astar_multi(maze)
    return path

def manhattan_distance(dot_1 , dot_2):
    return abs(dot_1[0] - dot_2[0]) + abs(dot_1[1] - dot_2[1])

def heuristic(remaining_dots):
    """
    Computes the Minimum Spanning Tree (MST) heuristic using Kruskal's algorithm.
    
    @param remaining_dots: List of remaining dot positions.

    @return mst_cost: The minimum spanning tree cost (heuristic value).
    """
    if len(remaining_dots) == 0:
        return 0

    dots = sorted(remaining_dots)

    edges = []
    for i in range(len(dots)):
        for j in range(i + 1, len(dots)):
            d1, d2 = dots[i], dots[j]
            dist = manhattan_distance(d1 , d2)
            edges.append((dist, d1, d2))

    edges.sort()
  
    parent = {}  
    for dot in dots:  
        parent[dot] = dot  

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x
            return True
        return False

    mst_cost = 0
    for dist, d1, d2 in edges:
        if union(d1, d2):  
            mst_cost += dist

    return mst_cost

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()  
    goals = set(maze.getObjectives()) 
    queue = [(0, 0, start, [start], frozenset())]  
    visited = set()

    while len(queue) > 0:
        print(queue)
        f , g, current, path, collected = heapq.heappop(queue)
        print(f)
        if collected == goals:
            return path

        state = (current, collected)
        if state in visited:
            continue
        visited.add(state)

        # Explore neighbors
        for neighbor in maze.getNeighbors(*current):
            new_collected = collected | ({neighbor} & goals)
            h = heuristic(goals - new_collected) 
            f_new = g + 1 + h  
            heapq.heappush(queue, (f_new , g + 1, neighbor, path + [neighbor], new_collected))

    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    goals = set(maze.getObjectives())
    full_path = []
    current_position = start

    while len(goals) > 0:
        # Greedy Best-First Search (GBFS) for the nearest goal
        queue = []
        heapq.heappush(queue, (0, current_position, [current_position]))  # (h-score, position, current path)
        visited = set()
        found_goal = None  # Track the goal reached

        while len(queue) > 0:
            empty , current, path = heapq.heappop(queue)

            if current in goals:
                found_goal = current
                full_path.extend(path[:-1])  # Append path except last step to avoid duplicates
                current_position = current
                break  # Stop search and move to the next goal

            if current in visited:
                continue
            visited.add(current)

            for neighbor in maze.getNeighbors(*current):
                if neighbor not in visited:
                    h_score = min(manhattan_distance(neighbor, goal) for goal in goals)  # Only heuristic, no g-score
                    heapq.heappush(queue, (h_score, neighbor, path + [neighbor]))

        if found_goal:
            goals.remove(found_goal)  # Remove the reached goal
            full_path.append(found_goal)  # Ensure goal is included in the path
        else:
            return []  # If a goal is unreachable, return failure

    return full_path
