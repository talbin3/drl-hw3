from GridWorld import GridWorld
from DynaQPlus import DynaQPlus

def main():
    # experiment settings
    numRows = 5
    numCols = 5
    gridWorld = GridWorld(numRows, numCols, (0, 0), (1, 1))

    # run experiment
    n = 10
    epsilon = 0.1
    stepSize = 0.01
    dynaQP = DynaQPlus(n, epsilon, stepSize, gridWorld)
    dynaQP.learn()




if __name__ == "__main__":
    main()