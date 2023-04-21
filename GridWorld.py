from enum import Enum

class Cell(Enum):
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
        self.map = [[Cell.OPEN] * rows]  * cols


    # take provided action in current state; return resulting state information
    def step(self, action):
        # up-action
        if action == 0: 
            # bounds check 
            if self.currentPos[0] == 0:
                return 0, self.currentPos, False
            
            # not at boundary; make move 
            self.currentPos = (self.currentPos[0] - 1, self.currentPos[1]) 
             
        # right-action
        if action == 1:
            # bounds check 
            if self.currentPos[1] == self.cols - 1:
                return 0, self.currentPos, False
            
            # not at boundary; make move 
            self.currentPos = (self.currentPos[0], self.currentPos[1] + 1) 

        # down-action
        if action == 2: 
            # bounds check 
            if self.currentPos[0] == self.rows - 1:
                return 0, self.currentPos, False
            
            # not at boundary; make move 
            self.currentPos = (self.currentPos[0] + 1, self.currentPos[1]) 

        # left-action
        if action == 3:
            # bounds check 
            if self.currentPos[1] == 0:
                return 0, self.currentPos, False
            
            # not at boundary; make move 
            self.currentPos = (self.currentPos[0], self.currentPos[1] - 1) 

        # report results
        reward = 1 if self.currentPos == self.goal else 0
        done = True if self.currentPos == self.goal else False
        return reward, self.currentPos, done

             
    def reset(self):
        self.currentPos = self.start