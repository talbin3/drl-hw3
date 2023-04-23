from GridWorld import GridWorld
from DynaQPlus import DynaQPlus

def main():
    # experiment settings
    numRows = 12
    numCols = 12
    gridWorld = GridWorld(numRows, numCols, (0, 0), (11, 11))

    # run experiment
    numEpisodes = 200
    n = 5
    epsilon = 0.01
    stepSize = 0.1
    kappa = 0.001
    dynaQP = DynaQPlus(numEpisodes, n, epsilon, stepSize, kappa, gridWorld)

    dynaQP.learn()
    dynaQP.plot()
    print(dynaQP.q)



if __name__ == "__main__":
    main()
