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
        numEpisodes = 50
        epNum = 0
        for i in range(numEpisodes):
            # reset gridworld for next episode
            self.gridWorld.reset()

            # track whether current episode has terminated
            done = False

            # number of steps taken during current episode (performance measure)
            stepCount = 0

            # track current state of current episode 
            currentPos = self.gridWorld.start

            while not done: 
                # store current state for state-action-value computation
                s = self.getIndex(currentPos)

                # select action (epsilon-greedy)
                action = self.selectAction(s)

                # take the action; observe R, S'
                reward, newPos, done = self.gridWorld.step(action)

                if reward == 1:
                    print("here")


                # update position
                currentPos = newPos

                # direct RL update
                sNew = self.getIndex(newPos) # new state
                self.q[s][action] = self.q[s][action] + self.stepSize * (reward + (self.discountRate * self.getBestActionValue(sNew)) - self.q[s][action])

                # model update
                self.model[s][action] = (reward, sNew)

                stepCount += 1

                # TODO: planning


            print("EP " + str(epNum) + " COMPLETE: " + str(stepCount))
            stepCount = 0 # reset for next iter

            epNum += 1
            print(self.q)

            
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


        


