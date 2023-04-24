from enum import Enum
import numpy as np

OPEN = 0
BLOCKED = 1
START = 2
GOAL = 3

class GridWorld:
    def __init__(self, rows, cols, start, goal):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.currentPos = start

        # create map
        self.map = np.full((rows, cols), 0)


    # take provided action in current state; return resulting state information
    def step(self, action):
        # up-action
        if action == 0: 
            # bounds check 
            if self.currentPos[0] == 0 or self.map[self.currentPos[0] - 1][self.currentPos[1]] == BLOCKED:
                return 0, self.currentPos, False
            
            # not at boundary; make move 
            self.currentPos = (self.currentPos[0] - 1, self.currentPos[1]) 
             
        # right-action
        if action == 1:
            # bounds check 
            if self.currentPos[1] == self.cols - 1 or self.map[self.currentPos[0]][self.currentPos[1] + 1] == BLOCKED:
                return 0, self.currentPos, False
            
            # not at boundary; make move 
            self.currentPos = (self.currentPos[0], self.currentPos[1] + 1) 

        # down-action
        if action == 2: 
            # bounds check 
            if self.currentPos[0] == self.rows - 1 or self.map[self.currentPos[0] + 1][self.currentPos[1]] == BLOCKED:
                return 0, self.currentPos, False
            
            # not at boundary; make move 
            self.currentPos = (self.currentPos[0] + 1, self.currentPos[1]) 

        # left-action
        if action == 3:
            # bounds check 
            if self.currentPos[1] == 0 or self.map[self.currentPos[0]][self.currentPos[1] - 1] == BLOCKED:
                return 0, self.currentPos, False
            
            # not at boundary; make move 
            self.currentPos = (self.currentPos[0], self.currentPos[1] - 1) 

        # report results
        reward = 1 if self.currentPos == self.goal else 0
        done = True if self.currentPos == self.goal else False
        return reward, self.currentPos, done


    # set map to map specified by input file (MAP DIMENSIONS AND SELF.ROWS/SELF.COLS MUST MATCH)
    def buildMap(self, filename):
        # read input map
        with open(filename) as f:
            lines = f.readlines()

        for i in range(self.rows):
            for j in range(self.cols):
                self.map[i][j] = self.getCell(lines[i][j])

                # set start and goal positions 
                if self.map[i][j] == START:
                    self.start = (i, j)
                elif self.map[i][j] == GOAL:
                    self.goal = (i, j)


    def reset(self):
        self.currentPos = self.start


    def getCell(self, char):
        #print(char)
        if char == 'O':
            return OPEN
        elif char == 'B':
            return BLOCKED
        elif char == 'G':
            return GOAL
        else: 
            return START