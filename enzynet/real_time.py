"""Plot in real-time."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>
# MIT License

from typing import Optional, Text

import time

from collections import deque
from matplotlib import pyplot as plt

start = time.time()


# Adapted from https://gist.github.com/Uberi/283a13b8a71a46fb4dc8.
class RealTimePlot(object):
    def __init__(self, max_entries: int = 200, x_label: Text = r'Epochs',
                 y_label: Text = r'Accuracy') -> None:
        # TeX friendly.
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        # Store.
        self.fig, self.axes = plt.subplots()
        self.max_entries = max_entries

        # x-axis.
        self.axis_x = deque(maxlen=max_entries)

        # Training accuracy.
        self.axis_y_tr = deque(maxlen=max_entries)
        self.lineplot_tr, = self.axes.plot([], [], "ro-")

        # Validation accuracy.
        self.axis_y_val = deque(maxlen=max_entries)
        self.lineplot_val, = self.axes.plot([], [], "bo-")

        # Autoscale.
        self.axes.set_autoscaley_on(True)

        # Set label names.
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

    def add(self, x: float, y_tr: float, y_val: Optional[float] = None) -> None:
        # Add new point.
        self.axis_x.append(x)
        self.axis_y_tr.append(y_tr)
        self.lineplot_tr.set_data(self.axis_x, self.axis_y_tr)

        if y_val != None:  # Validation accuracy is specified.
            self.axis_y_val.append(y_val)
            self.lineplot_val.set_data(self.axis_x, self.axis_y_val)

        # Change axis limits.
        self.axes.set_xlim(self.axis_x[0], self.axis_x[-1] + 1e-15)
        self.axes.relim(); self.axes.autoscale_view()  # Rescale the y-axis.


if __name__ == "__main__":
    # Initialization.
    display = RealTimePlot(max_entries=100)
    i = 0

    # Update in real-time.
    while True:
        display.add(time.time() - start, i, i/2)
        plt.pause(5)
        i = i+1
