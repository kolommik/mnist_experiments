""" Module plot_graph contains diffrent visualization procedures """

import numpy as np
import matplotlib.pyplot as plt


def plot_digits(*args: np.ndarray):
    """Plots pictures of the digits of a datasets as visual array.

    Parameters:
        *args: (np.ndarray) - tuple of np.ndarray(number_of_digits_to_show, height, width, 1)
    """
    args = [x.squeeze() for x in args]
    rows_num = len(args)
    cols_num = min([x.shape[0] for x in args])
    plt.figure(figsize=(2 * cols_num, 2 * rows_num))
    for j in range(cols_num):
        for i in range(rows_num):
            ax = plt.subplot(rows_num, cols_num, i * cols_num + j + 1)
            ax.imshow(args[i][j], cmap=plt.cm.binary)
            ax.axis("off")
    plt.show()


def plot_digits_page(
    digits_arr: np.array,
    reshape_pixels: int = 28,
    title: str = "A selection from the 784-dimensional digits dataset",
):
    """Plots 10x10 pictures of the digits datasets array.

    Parameters:
        * digits_arr: np.ndarray - array of digits(digits, height, width, 1)
        * reshape_pixels: int - reshape ratio, default 28x28
        * title: str - title for picture
    """
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
    for idx, ax in enumerate(axs.ravel()):
        ax.imshow(
            digits_arr[idx].reshape((reshape_pixels, reshape_pixels)),
            cmap=plt.cm.binary,
        )
        ax.axis("off")
    _ = fig.suptitle(title, fontsize=16)
    plt.show()


# a = np.random.rand(10, 28, 28,1) * 10
# print(a.shape)
# plot_digits(a)
