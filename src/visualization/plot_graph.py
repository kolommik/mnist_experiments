""" Module plot_graph contains diffrent visualization procedures """

from typing import Optional, Tuple, List
import io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
from scipy.stats import norm

# Нарисовать цифры из нескольких массивов
def plot_digits(*args: np.ndarray, show: bool = True) -> mpl.figure.Figure:
    """Plots pictures of the digits of a datasets as visual array.

    Parameters
    ----------
    *args : np.ndarray
        Tuple of np.ndarray, with the shape
        (number_of_digits_to_show, height, width, 1)
    show : bool, optional
        Whether to show the resulting digit manifold using matplotlib,
        by default True.

    Returns
    -------
    figure : Matplotlib Figure
        A Matplotlib Figure object of the scatter plot.

    """
    args = [x.squeeze() for x in args]
    rows_num = len(args)
    cols_num = min([x.shape[0] for x in args])
    fig = plt.figure(figsize=(2 * cols_num, 2 * rows_num))
    for j in range(cols_num):
        for i in range(rows_num):
            ax = plt.subplot(rows_num, cols_num, i * cols_num + j + 1)
            ax.imshow(args[i][j], cmap=plt.cm.binary)
            ax.axis("off")
    if show:
        plt.show()
    else:
        plt.close()

    return fig


# Нарисовать цифры 10х10
def plot_digits_page(
    digits_arr: np.array,
    reshape_pixels: int = 28,
    title: str = "A selection from the 784-dimensional digits dataset",
    show: bool = True,
) -> mpl.figure.Figure:
    """Plots 10x10 pictures of the digits datasets array.

    Parameters
    ----------
    digits_arr: np.ndarray
        array of digits(digits, height, width, 1)
    reshape_pixels: int
        reshape ratio, default 28x28
    title: str
        title for picture
    show : bool, optional
        Whether to show the resulting digit manifold using matplotlib,
        by default True.

    Returns
    -------
    figure : Matplotlib Figure
        A Matplotlib Figure object of the scatter plot.

    """
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
    for idx, ax in enumerate(axs.ravel()):
        ax.imshow(
            digits_arr[idx].reshape((reshape_pixels, reshape_pixels)),
            cmap=plt.cm.binary,
        )
        ax.axis("off")
    _ = fig.suptitle(title, fontsize=16)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# Гомотопия по прямой в пространстве объектов или в пространстве кодов
def plot_homotopy(frm, to, n=10, decoder=None) -> None:
    """Plots homotopy between 2 objects

    Parameters
    ----------
        * frm - object FROM what we started
        * to - object TO what we going
        * n=10 - number of between objects
        * decoder=None - decoder, if it's coded shapes
    """
    # разбиваем интервал от 0 до 1 на n шагов
    z = np.zeros(([n] + list(frm.shape)))

    # в цикле пробегаемся по этим шагам
    # при этом делаем морфинг объекта frm к объекту to
    # постепенно уменьшая вес объекта frm,
    # в то же время пропорционально увеличивая вес объекта to
    # если указан декодер - декодируем полученные объекты
    for i, t in enumerate(np.linspace(0.0, 1.0, n)):
        z[i] = frm * (1 - t) + to * t
    if decoder:
        plot_digits(decoder.predict(z, batch_size=n))
    else:
        plot_digits(z)


# Решетка, где узлы нормально распледелены в 2д
def sapmpling_2d_normal_grid(n_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sampling 2D Normal Grid.

    Parameters
    ----------
    n_count : int
        The number of samples to generate.

    Returns
    -------
    grid_x, grid_y : ndarray
        The x and y coordinates of the grid nodes.

    Notes
    -----
    This function generates a grid of nodes for sampling from a 2D
    normal distribution N(0, I).
    The grid is generated from the inverse cumulative distribution
    function of the normal distribution.
    """
    # Так как сэмплируем из N(0, I),
    # то сетку узлов, в которых генерируем цифры
    # берем из обратной функции распределения
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n_count))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n_count))
    return grid_x, grid_y


# Диаграмма рассеяния 2д латентного пространства
def plot_2d_latent_space_grid(
    digit_pics: np.ndarray,
    labels: np.ndarray,
    encoded_values: np.ndarray,
    show: bool = True,
    plot_samples: bool = False,
    plot_grid: bool = False,
    grid_x: Optional[np.ndarray] = None,
    grid_y: Optional[np.ndarray] = None,
    show_colorbar: bool = True,
) -> mpl.figure.Figure:
    """Plots a 2D scatter plot of the latent space encoding of digit
    images and an optional grid, and/or samples of digit images.

    Parameters
    ----------
    digit_pics : array_like
        An array of digit images.
    labels : array_like
        An array of labels corresponding to each digit image.
    encoded_values : array_like
        An array of 2D latent space encodings of digit images.
    show : bool, optional
        Whether to show the plot or not. Default is True.
    plot_samples : bool, optional
        Whether to plot samples of digit images. Default is False.
    plot_grid : bool, optional
        Whether to plot a grid in the scatter plot. Default is False.
    grid_x : array_like, optional
        An array of x-coordinates for the grid.
    grid_y : array_like, optional
        An array of y-coordinates for the grid.
    show_colorbar: bool, optional
        Whether to show the colorbar on the left or not. Default is True.
    Returns
    -------
    figure : Matplotlib Figure
        A Matplotlib Figure object of the scatter plot.
    """
    # create figure
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.tab10
    plt.scatter(encoded_values[:, 0], encoded_values[:, 1], c=labels, s=5, cmap=cmap)

    # plot grid_x and grid_y
    if plot_grid is True:
        for x in grid_x:
            plt.axvline(x=x, color="grey", alpha=0.8, linewidth=0.7)

        for y in grid_y:
            plt.axhline(y=y, color="grey", alpha=0.8, linewidth=0.7)

    # plot boxes with samples
    if plot_samples:
        image_positions = np.array([[1.0, 1.0]])
        for index, position in enumerate(encoded_values):
            dist = np.sum((position - image_positions) ** 2, axis=1)
            if np.min(dist) > 2:  # if far enough from other images
                image_positions = np.r_[image_positions, [position]]
                imagebox = mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(
                        digit_pics[index], cmap="binary", zoom=0.4
                    ),
                    position,
                    bboxprops={"edgecolor": cmap(labels[index]), "lw": 2},
                )
                plt.gca().add_artist(imagebox)

    # add color mapping bar to the right
    if show_colorbar:
        plt.colorbar()

    # scale and center picture
    max_value = encoded_values.max()
    plt.xlim(-1.1 * max_value, 1.1 * max_value)
    plt.ylim(-1.1 * max_value, 1.1 * max_value)

    # add horizontal and vertical lines
    plt.axhline(y=0, color="red", alpha=0.7, linewidth=1, linestyle="--")
    plt.axvline(x=0, color="red", alpha=0.7, linewidth=1, linestyle="--")

    # create figure from it for return
    figure = plt.gcf()

    # show or not to show
    if show:
        plt.show()
    else:
        plt.close()

    return figure


# Рисует мноообразие цифр для каждого узла сетки в латентн.простр-ве
def draw_manifold(
    generator,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    show: bool = True,
    digit_size: int = 28,
) -> mpl.figure.Figure:
    """Draw a digit manifold using the provided generator.

    Parameters
    ----------
    generator : generator model
        The generator to use for producing digits.
    grid_x, grid_y : np.ndarray
        The x and y coordinates of the grid nodes to generate digits at.
    show : bool, optional
        Whether to show the resulting digit manifold using matplotlib,
        by default True.
    digit_size : int, optional
        The size of each digit in the manifold, by default 28.

    Returns
    -------
    figure : Matplotlib Figure
        A Matplotlib Figure object of the scatter plot.

    Notes
    -----
    The digit manifold is generated by drawing a digit at each grid node
    and concatenating them.
    """
    # Рисование цифр из 2d многообразия
    latent_dim = 2
    figure = np.zeros((digit_size * len(grid_x), digit_size * len(grid_y)))
    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y[::-1]):

            z_sample = np.zeros((1, latent_dim))
            z_sample[:, :2] = np.array([[xi, yi]])

            x_decoded = generator.predict(z_sample, verbose=0)
            digit = x_decoded[0].squeeze()
            figure[
                j * digit_size : (j + 1) * digit_size,  # noqa: E203
                i * digit_size : (i + 1) * digit_size,  # noqa: E203
            ] = digit
    # Визуализация
    plt.figure(figsize=(7, 7))
    plt.imshow(figure, cmap="Greys")
    plt.grid(None)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    figure = plt.gcf()

    if show:
        plt.show()
    else:
        plt.close()

    return figure


# Рисует мноообразие цифр для каждого узла сетки в латентн.простр-ве
def draw_manifold_cvae(
    generator,
    lbl: int,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    show: bool = True,
    digit_size: int = 28,
) -> mpl.figure.Figure:
    """Draw a digit manifold using the provided generator for CVAE.

    Parameters
    ----------
    generator : generator model
        The CVAE generator to use for producing digits.
    lbl: int
        Label for CVAE generator
    grid_x, grid_y : np.ndarray
        The x and y coordinates of the grid nodes to generate digits at.
    show : bool, optional
        Whether to show the resulting digit manifold using matplotlib,
        by default True.
    digit_size : int, optional
        The size of each digit in the manifold, by default 28.

    Returns
    -------
    figure : Matplotlib Figure
        A Matplotlib Figure object of the scatter plot.

    Notes
    -----
    The digit manifold is generated by drawing a digit at each grid node
    and concatenating them.
    """
    # Рисование цифр из 2d многообразия
    latent_dim = 2

    input_lbl = np.zeros((1, 10))
    input_lbl[0, lbl] = 1

    figure = np.zeros((digit_size * len(grid_x), digit_size * len(grid_y)))
    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y[::-1]):

            z_sample = np.zeros((1, latent_dim))
            z_sample[:, :2] = np.array([[xi, yi]])

            x_decoded = generator.predict([z_sample, input_lbl], verbose=0)
            digit = x_decoded[0].squeeze()
            figure[
                j * digit_size : (j + 1) * digit_size,  # noqa: E203
                i * digit_size : (i + 1) * digit_size,  # noqa: E203
            ] = digit
    # Визуализация
    plt.figure(figsize=(7, 7))
    plt.imshow(figure, cmap="Greys")
    plt.grid(None)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    figure = plt.gcf()

    if show:
        plt.show()
    else:
        plt.close()

    return figure


# Переводит mpl.Figure в массив-картинку
def convert_figures_to_array(fig: mpl.figure.Figure) -> np.array:
    """Convert a Matplotlib figure to a NumPy array.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The Matplotlib figure to be converted to an array.

    Returns
    -------
    array : `numpy.ndarray`
        The NumPy array representation of the figure.

    Notes
    -----
    This function uses `BytesIO` from the `io` module to save the figure to a buffer
    in PNG format, and then uses `mpimg.imread` to read the buffer and return a
    NumPy array. The figure is saved with 'tight' bounds.
    """
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        array = mpimg.imread(buf)
    return array


# Объединяет в 1 картинку две картинки в формате mpl.Figure
def combine_two_figures_horisontally(
    fig1: mpl.figure.Figure,
    fig2: mpl.figure.Figure,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
) -> mpl.figure.Figure:
    """Combine two matplotlib figures horizontally.

    Parameters
    ----------
    fig1 : mpl.figure.Figure
        The first figure to be combined.
    fig2 : mpl.figure.Figure
        The second figure to be combined.
    figsize : Tuple[int, int]
        The size of the final combined figure. Default is (15, 7).
    show : bool, optional
            Whether to show the resulting digit manifold using matplotlib,
            by default True.
    Returns
    -------
    figure : Matplotlib Figure
        The combined figure.
    """

    arr1 = convert_figures_to_array(fig1)
    arr2 = convert_figures_to_array(fig2)

    # Combine the figures side by side
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.imshow(arr1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.imshow(arr2)

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    if show:
        plt.show()
    else:
        plt.close()
    return fig


# Создает анимацию из массива картинок mpl.figure.Figure
def create_animation_from_array_of_figures(
    figures_list: List[mpl.figure.Figure],
    titles_list: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    filename: Optional[str] = None,
) -> mpl.animation.FuncAnimation:
    """Create an animation from an array of figures.

    Parameters
    ----------
    figures_list : list
        List of figures to be animated.
    titles_list : list
        List of titles for each frame of the animation.
    figsize : tuple, optional
        Tuple representing the size of the figure, by default None
    filename : str, optional
        File name to save the animation, by default None

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation created from the input figures.

    """

    def update(frame):
        plt.clf()  # Clear the previous plot
        arr1 = convert_figures_to_array(figures_list[frame])
        # Recreate the plot with new data
        plt.imshow(arr1)
        plt.axis("off")
        plt.title(str(titles_list[frame]))
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    # canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig)
    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(figures_list)), interval=250
    )

    # Save the animation as an MP4 file
    if filename is not None:
        # writer = animation.FFMpegWriter(fps=4, bitrate=1800)
        writer = animation.FFMpegWriter(fps=4, bitrate=500)
        ani.save(filename, writer=writer)

    plt.close(fig)
    return ani
