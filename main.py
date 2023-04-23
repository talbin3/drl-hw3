from GridWorld import GridWorld
from DynaQPlus import DynaQPlus

def main():
    # experiment settings
    numRows = 12
    numCols = 12
    gridWorld = GridWorld(numRows, numCols, (0, 0), (11, 11))

    # run experiment
    n = 1
    epsilon = 0.05
    stepSize = 0.1
    dynaQP = DynaQPlus(n, epsilon, stepSize, gridWorld)

    dynaQP.learn()
    #print(dynaQP.q)
    dynaQP.plot()



if __name__ == "__main__":
    main()
