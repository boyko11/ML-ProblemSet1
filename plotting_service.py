import numpy as np
import matplotlib.pyplot as plt

def plot_simple_line_chart(x_values, y_values, x_label, y_label, title):

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()

    plt.plot(x_values, y_values)

    plt.legend(loc="best")
    #plt.ylim(0.96, 1.00)
    plt.show()
    #plt.savefig('ProjectionLoss-{0}-{1}.png'.format(algo, dataset))


def plot_scatter_plot_and_best_fit_line(x_values, y_values, x_label, y_label, title, slope, intercept):

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()

    #plt.plot(x_values, y_values)
    plt.scatter(x_values, y_values)

    x = np.linspace(np.min(np.asarray(x_values)) - 2, np.max(np.asarray(x_values)) + 2, 10)
    y = slope*x + intercept
    plt.plot(x, y)

    plt.legend(loc="best")
    #plt.ylim(0.96, 1.00)
    plt.show()
    #plt.savefig('ProjectionLoss-{0}-{1}.png'.format(algo, dataset))

