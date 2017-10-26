import numpy as np
import matplotlib
matplotlib.use('AGG') # make matplotlib work on aws
import matplotlib.pyplot as plt
import PIL
import os
import scipy


# ==============================================================================
#                                                        BATCH_OF_IMAGES_TO_GRID
# ==============================================================================
# Code Taken from: https://github.com/ronrest/convenience_py/blob/master/ml/img/batch_of_images_to_grid.py
def batch_of_images_to_grid(imgs, rows, cols):
    """
    Given a batch of images stored as a numpy array of shape:

           [n_batch, img_height, img_width]
        or [n_batch, img_height, img_width, n_channels]

    it creates a grid of those images of shape described in `rows` and `cols`.

    Args:
        imgs: (numpy array)
            Shape should be either:
                - [n_batch, im_rows, im_cols]
                - [n_batch, im_rows, im_cols, n_channels]

        rows: (int) How many rows of images to use
        cols: (int) How many cols of images to use

    Returns: (numpy array)
        The grid of images as one large image of either shape:
            - [n_classes*im_cols, num_per_class*im_rows]
            - [n_classes*im_cols, num_per_class*im_rows, n_channels]
    """
    # TODO: have a resize option to rescale the individual sample images
    # TODO: Have a random shuffle option
    # TODO: Set the random seed if needed
    # if seed is not None:
    #     np.random.seed(seed=seed)

    # Only use the number of images needed to fill grid
    assert rows>0 and cols>0, "rows and cols must be positive integers"
    n_cells = (rows*cols)
    imgs = imgs[:n_cells]

    # Image dimensions
    n_dims = imgs.ndim
    assert n_dims==3 or n_dims==4, "Incorrect # of dimensions for input array"

    # Deal with images that have no color channel
    if n_dims == 3:
        imgs = np.expand_dims(imgs, axis=3)

    n_batch, img_height, img_width, n_channels = imgs.shape

    # Handle case where there is not enough images in batch to fill grid
    n_gap = n_cells - n_batch
    imgs = np.pad(imgs, pad_width=[(0,n_gap),(0,0), (0,0), (0,0)], mode="constant", constant_values=0)

    # Reshape into grid
    grid = imgs.reshape(rows,cols,img_height,img_width,n_channels).swapaxes(1,2)
    grid = grid.reshape(rows*img_height,cols*img_width,n_channels)

    # If input was flat images with no color channels, then flatten the output
    if n_dims == 3:
        grid = grid.squeeze(axis=2) # axis 2 because batch dim has been removed

    return grid


# ==============================================================================
#                                                                       SHOW_IMG
# ==============================================================================
def show_img(a):
    """Given a numpy array representing an image, view it"""
    img = PIL.Image.fromarray(a)
    img.show()

# ==============================================================================
#                                                                   TRAIN_CURVES
# ==============================================================================
# Code taken from: https://github.com/ronrest/convenience_py/blob/master/ml/plots/train_curves.py
def train_curves(train, valid, saveto=None, title="Accuracy over time", ylab="accuracy", legend_pos="lower right"):
    """ Plots the training curves. If `saveto` is specified, it saves the
        the plot image to a file.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(title, fontsize=15)
    ax.plot(train, color="#FF4F40",  label="train")
    ax.plot(valid, color="#307EC7",  label="valid")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylab)

    # Grid lines
    ax.grid(True)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#888888', linestyle='-')
    plt.grid(b=True, which='minor', color='#AAAAAA', linestyle='-', alpha=0.2)

    # Legend
    ax.legend(loc=legend_pos, title="", frameon=False,  fontsize=8)

    # Save or show
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)
        plt.close()
