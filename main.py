from GridWorld import GridWorld
from DynaQPlus import DynaQPlus
from DynaQ import DynaQ, DynaQType
import matplotlib.pylab as plt

def main():
    # experiment settings
    numRows = 12
    numCols = 12
    gridWorld = GridWorld(numRows, numCols, (0, 0), (11, 11))

    # run experiment
    type = DynaQType.NORMAL
    numEpisodes = 200
    n = 5
    epsilon = 0.01
    stepSize = 0.1
    kappa = 0.001

    # Normal DynaQ (no exploration bonus)
    dynaQ = DynaQ(type, numEpisodes, n, epsilon, stepSize, kappa, gridWorld)
    dynaQ.learn()

    # DynaQ+ with updates
    type = DynaQType.DYNA_Q_PLUS
    dynaQP = DynaQ(type, numEpisodes, n, epsilon, stepSize, kappa, gridWorld)
    dynaQP.learn()

    # DynaQ+ NO UPDATE
    type = DynaQType.DYNA_Q_PLUS_NO_UPDATE
    dynaQPNoUpdate = DynaQ(type, numEpisodes, n, epsilon, stepSize, kappa, gridWorld)
    dynaQPNoUpdate.learn()

    plotResults(dynaQ, dynaQP, dynaQPNoUpdate)


def plotResults(dynaQ, dynaQP, dynaQPNoUpdate):
    fig, ax = plt.subplots()
    ax.plot(dynaQ.cumulativeRewards, color='r')
    ax.plot(dynaQP.cumulativeRewards, color='b')
    ax.plot(dynaQPNoUpdate.cumulativeRewards, color='g')
    plt.show()


if __name__ == "__main__":
    main()
