from GridWorld import GridWorld
import random

class DynaQPlus:
    def __init__(self, n, epsilon, stepSize, gridWorld):
        self.n = n                                              # number of planning iterations
        self.epsilon = epsilon
        self.stepSize = stepSize
        self.discountRate = 0.95
        self.gridWorld = gridWorld                              # gridWorld instance to be trained on
        self.q = [[0] * 4] * (gridWorld.rows * gridWorld.cols)  # state-value function          
        self.model = [[(0, 0)] * 4] * (gridWorld.rows * gridWorld.cols)

    def learn(self):
        numEpisodes = 10
        for i in range(numEpisodes):
            # reset gridworld for next episode
            self.gridWorld.reset()

            # track whether current episode has terminated
            done = False

            # track current state of current episode and use p
            currentPos = self.gridWorld.start

            while not done: 
                # store current state for state-action-value computation
                s = self.getIndex(currentPos)

                # select action (epsilon-greedy)
                action = self.selectAction(currentPos)

                # take the action; observe R, S'
                reward, newPos, done = self.gridWorld.step(action)

                # direct RL update
                sNew = self.getIndex(newPos) # new state
                print(sNew)
                self.q[s][action] = self.q[s][action] + self.stepSize * (reward + self.discountRate * self.getBestActionValue(sNew) - self.q[s][action])

                # model update
                self.model[s][action] = (reward, sNew)

                # planning
                for i in range(self.n):
                    pass

            
    def selectAction(self, s):
        # choose random action epsilon percent of the time
        if self.getDecision(self.epsilon):
            return random.randint(0, 3)     
        
        # choose greedy otherwise
        else: 
            return self.q[s].index(max(self.q[s]))
        

    # returns true (probability * 100)% of the time, false rest of time
    def getDecision(self, probability):
        return random.random() < probability
    
    
    # convert grid coordinates to state index
    def getIndex(self, coords):
        return (coords[0] * self.gridWorld.rows) + coords[1]
    

    # get value of best action in state s
    def getBestActionValue(self, s):
        return self.q[s].index(max(self.q[s]))


        


