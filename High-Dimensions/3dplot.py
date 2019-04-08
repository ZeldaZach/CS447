'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''
import pathlib
import re

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

if __name__ == "__main__":

    with pathlib.Path("hypersphere_output.txt").open("r") as f:
        output_content = f.read()

    content = re.split(r"Dimension [0-9]* Distance from center histogram", output_content)[1:]

    for dimension, value in enumerate(content):
        dimensions = []
        distances = []
        percentages = []
        rows = value.split("\n")
        for row in rows:
            v_split = row.split(": ")
            if len(v_split) == 2:
                distance, percentage = v_split[0].strip(), v_split[1][:-1].strip()
                distances.append(distance)
                percentages.append(percentage)
                dimensions.append(dimension+2)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.array(distances)
        Y = np.array(percentages)
        Z = np.array(dimensions)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(np.square(X) + np.square(Y))
        Z = np.sin(R)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()