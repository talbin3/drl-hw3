from GridWorld import GridWorld
from DynaQPlus import DynaQPlus

def main():
    # experiment settings
    numRows = 5
    numCols = 5
    gridWorld = GridWorld(numRows, numCols, (0, 0), (4, 4))

    # run experiment
    n = 1
    epsilon = 0.05
    stepSize = 0.1
    dynaQP = DynaQPlus(n, epsilon, stepSize, gridWorld)

    dynaQP.learn()
    dynaQP.plot()
    print(dynaQP.q)



if __name__ == "__main__":
    main()
