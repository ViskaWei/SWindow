import matplotlib.pyplot as plt

def plot_error(out):
    plt.scatter(np.log10(out.m), out['sketchErr'], label = 'sketch')
    plt.scatter(np.log10(out.m), out['uniformErr'], label = 'uniform')

    plt.legend()
    plt.ylabel('error')
    plt.xlabel('log stream size m')