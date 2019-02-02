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
