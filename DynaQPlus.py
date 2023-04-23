from GridWorld import GridWorld
import numpy as np
import random
import matplotlib.pylab as plt

class DynaQPlus:
    def __init__(self, n, epsilon, stepSize, gridWorld):
        self.n = n                                              # number of planning iterations
        self.epsilon = epsilon
        self.stepSize = stepSize
        self.discountRate = 0.95
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


        self.avgStepNum = 0
        self.stepsPerEp = []

    def learn(self):
        # store observed states and actions taken within them; if visited[s] = [False, False, False, False], s not visited yet
        visited = []
        for i in range(self.gridWorld.rows * self.gridWorld.cols):
            actionsTaken = np.array([False, False, False, False])
            visited.append(actionsTaken)


        numEpisodes = 200
        epNum = 0
        for i in range(numEpisodes):
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
               # print(action)

                # record action taken at s
                visited[s][action] = True 
                
                # take the action; observe R, S'
                reward, newPos, done = self.gridWorld.step(action)

                # update position
                currentPos = newPos

                # direct RL update
                sNew = self.getIndex(newPos) # new state
                self.q[s][action] = self.q[s][action] + self.stepSize * (reward + (self.discountRate * self.getBestActionValue(sNew)) - self.q[s][action])

                # model update
                self.model[s][action] = (reward, sNew)

                stepCount += 1

                # TODO: planning
                for j in range(self.n):
                    # get random (state, action) pair observed
                    (state, a) = self.getVisitedPair(visited)
                    result = self.model[state][a]
                    self.q[state][a] = self.q[state][a] + self.stepSize * (result[0] + (self.discountRate * self.getBestActionValue(result[1])) - self.q[state][a])




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
        a = np.argmax(self.q[s])
        #@print(a) 
        return a
    

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.stepsPerEp)
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




        


