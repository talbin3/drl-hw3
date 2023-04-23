from GridWorld import GridWorld
from DynaQPlus import DynaQPlus
import matplotlib.pylab as plt

def main():
    # experiment settings
    numRows = 12
    numCols = 12
    gridWorld = GridWorld(numRows, numCols, (0, 0), (11, 11))

    # run experiment
    modded = False
    numEpisodes = 200
    n = 5
    epsilon = 0.01
    stepSize = 0.1
    kappa = 0.001

    # un-modded DynaQ+
    dynaQP = DynaQPlus(modded, numEpisodes, n, epsilon, stepSize, kappa, gridWorld)
    dynaQP.learn()

    # modded DynaQ+
    modded = True
    dynaQPModded = DynaQPlus(modded, numEpisodes, n, epsilon, stepSize, kappa, gridWorld)
    dynaQPModded.learn()

    plotResults(dynaQP, dynaQPModded)


def plotResults(dynaQP, dynaQPModded):
    fig, ax = plt.subplots()
    ax.plot(dynaQP.cumulativeRewards, color='r')
    ax.plot(dynaQPModded.cumulativeRewards)
    plt.show()


if __name__ == "__main__":
    main()
