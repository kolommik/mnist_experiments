""" Module get_data contains data load and preprocessing procedures"""

from typing import Tuple
import numpy as np
import tensorflow as tf


def get_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads MNIST 28x28 grayscale images of the 10 digits data from keras,
    prepared for tf experiments.

    Returns:
      Tuple of NumPy arrays: `x_train, y_train, x_test, y_test`.
    **x_train** : float32 NumPy array of grayscale image data with shapes
      `(60000, 28, 28, 1)`, containing the training data. Pixel values range
      from 0 to 1.
    **y_train** : uint8 NumPy array of digit labels (integers in range 0-9)
      with shape `(60000,)` for the training data.
    **x_test** : float32 NumPy array of grayscale image data with shapes
      (10000, 28, 28, 1), containing the test data. Pixel values range
      from 0 to 255.
    **y_test** : uint8 NumPy array of digit labels (integers in range 0-9)
      with shape `(10000,)` for the test data.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    # Подготовка к TF(Keras)
    # нормализуем и получаем данные от 0 до 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # Решэйпим
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    return x_train, y_train, x_test, y_test
