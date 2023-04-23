from GridWorld import GridWorld
import numpy as np
import random
import matplotlib.pylab as plt
from enum import Enum

class DynaQType(Enum):
    NORMAL = 0
    DYNA_Q_PLUS = 1
    DYNA_Q_PLUS_NO_UPDATE = 2

class DynaQ:
    def __init__(self, type, numEpisodes, n, epsilon, stepSize, kappa, gridWorld):
        self.type = type
        self.numEpisodes = numEpisodes
        self.n = n                                              # number of planning iterations
        self.epsilon = epsilon
        self.stepSize = stepSize
        self.kappa = kappa
        self.discountRate = 0.5
        self.gridWorld = gridWorld                              # gridWorld instance to be trained on

        # allocate space for state-action value function 
        self.q = []
        for i in range(gridWorld.rows):
            for j in range(gridWorld.cols):
                actionSpace = np.array([0, 0, 0, 0], dtype=float)
                self.q.append(actionSpace)


        # allocate space for model 
        self.model = []
        for i in range(gridWorld.rows):
            for j in range(gridWorld.cols):
                inner = np.array([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], dtype=object)
                self.model.append(inner)

        # store avg cumulative return
        self.cumulativeRewards = [0]

        self.avgStepNum = 0
        self.stepsPerEp = []

        # store the number of timesteps since each (s, a) pair has been tried
        self.sinceLastVisit = []
        for i in range(self.gridWorld.rows * self.gridWorld.cols):
            timesTaken = np.array([0, 0, 0, 0])
            self.sinceLastVisit.append(timesTaken)

    def learn(self):
        # store observed states and actions taken within them; if visited[s] = [False, False, False, False], s not visited yet
        visited = []
        for i in range(self.gridWorld.rows * self.gridWorld.cols):
            actionsTaken = np.array([False, False, False, False])
            visited.append(actionsTaken)

        # store the number of timesteps since each (s, a) pair has been tried
        sinceLastVisit = []
        for i in range(self.gridWorld.rows * self.gridWorld.cols):
            timesTaken = np.array([0, 0, 0, 0])
            sinceLastVisit.append(timesTaken)

        cumulativeReward = 0
        epNum = 0
        for i in range(self.numEpisodes):
            print(i)
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

                # record action taken at s
                visited[s][action] = True 

                # add one to all (s, a) pairs sinceLastVisit, as a timestep has passed
                self.updateLastVisits(sinceLastVisit)
                sinceLastVisit[s][action] = 0
                
                # take the action; observe R, S'
                reward, newPos, done = self.gridWorld.step(action)

                # update cumulative reward
                cumulativeReward += reward
                self.cumulativeRewards.append(cumulativeReward)

                # update position
                currentPos = newPos

                # direct RL update
                sNew = self.getIndex(newPos) # new state
                val = self.q[s][action] + self.stepSize * (reward + (self.discountRate * self.getBestActionValue(sNew)) - self.q[s][action])
                self.q[s][action] = val

                # model update
                self.model[s][action] = (reward, sNew)

                stepCount += 1

                # planning
                for j in range(self.n):
                    # get random (state, action) pair observed
                    (state, a) = self.getVisitedPair(visited)

                    # simluate a transition
                    result = self.model[state][a]

                    # deterimine reward used in update based on DynaQ type
                    reward = self.getReward(result[0], state, a) 

                    #reward = result[0]
                    self.q[state][a] = self.q[state][a] + self.stepSize * (reward + (self.discountRate * self.getBestActionValue(result[1])) - self.q[state][a])



#            print("EP " + str(epNum) + " COMPLETE: " + str(stepCount))

            # update step-count average
            self.avgStepNum += (1 / (epNum + 1)) * (stepCount - self.avgStepNum)

            self.stepsPerEp.append(stepCount)

            stepCount = 0 # reset for next iter

            epNum += 1
        


            
    def selectAction(self, s):
        # choose random action epsilon percent of the time
        if self.getDecision(self.epsilon):
            return random.randint(0, 3)     
        
        # choose greedy otherwise
        else:
            return self.randomArgMax(self.q[s]) 
            #return np.argmax(self.q[s])
        

    # returns true (probability * 100)% of the time, false rest of time
    def getDecision(self, probability):
        return random.random() < probability
    
    
    # convert grid coordinates to state index
    def getIndex(self, coords):
        return (coords[0] * self.gridWorld.rows) + coords[1]
    

    # get value of best action in state s
    def getBestActionValue(self, s):
        # add in exploration bonuses to all actions, select from there (NOT CHANGING Q VALUE)
        if self.type == DynaQType.DYNA_Q_PLUS_NO_UPDATE: 
            aSpace = self.q[s].copy()
            for i in range(len(aSpace)):
                aSpace[i] = aSpace[i] + self.kappa * np.sqrt(self.sinceLastVisit[s][i])
            a = self.randomArgMax(aSpace)

        else:
            a = self.randomArgMax(self.q[s])
        return self.q[s][a]
    

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.cumulativeRewards)
        plt.show()


    # argmax of array a; ties broken randomly
    def randomArgMax(self, a):
        max = 0
        maxIndex = 0
        nohit = True
        for i in range(len(a)):
            if a[i] > max:
                nohit = False
                max = a[i]
                maxIndex = i

        if nohit:
            return random.randint(0, 3)     
        
        else: 
            return maxIndex


    def getVisitedPair(self, visited):
        done = False
        while not done: 
            s = random.randint(0, len(self.q) - 1)
            actionsTaken = []
            for i in range(4):
                if visited[s][i]:
                    actionsTaken.append(i)

            try: 
                return (s, random.choice(actionsTaken))

            except IndexError: # state not seen-select new random state
                continue


    def updateLastVisits(self, a):
        for i in range(len(a)):
            for j in range(len(a[i])):
                a[i][j] += 1

        

    def getReward(self, transitionReward, s, a):
        if self.type == DynaQType.DYNA_Q_PLUS: 
            return transitionReward + self.kappa * np.sqrt(self.sinceLastVisit[s][a])
        
        else:
            return transitionReward


        

