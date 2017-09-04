import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy


# ==============================================================================
#                                                        BATCH_OF_IMAGES_TO_GRID
# ==============================================================================
def batch_of_images_to_grid(imgs, rows, cols):
    """ Given a batch of images stored as a numpy array of shape:
            [n_batch, img_height, img_width, n_channels]
        it creates a grid of images of shape described in `rows` and `cols`.

    Args:
        imgs: (numpy array)
            Shape should be:
                - [n_batch, im_rows, im_cols, n_channels]

        rows: (int) How many rows of images to use
        cols: (int) How many cols of images to use

    Returns: (numpy array)
        The grid of images as one large image of shape:
            - [n_classes*im_cols, num_per_class*im_rows, n_channels]
    """
    # TODO: have a resize option to rescale the individual sample images
    # Set the random seed if needed

    # TODO: Have a random shuffle option
    # if seed is not None:
    #     np.random.seed(seed=seed)

    # Only use the number of images needed to fill grid
    n_cells = (rows*cols)
    imgs = imgs[:n_cells]

    n_batch, img_height, img_width, n_channels = imgs.shape

    # Handle case where there is not enough images in batch to fill grid
    n_gap = n_cells - n_batch
    imgs = np.pad(imgs, pad_width=[(0,n_gap),(0,0), (0,0), (0,0)], mode="constant", constant_values=0)

    # Reshape into grid
    grid = imgs.reshape(rows,cols,img_height,img_width,n_channels).swapaxes(1,2)
    grid = grid.reshape(rows*img_height,cols*img_width,n_channels)

    return grid


# ==============================================================================
#                                          GRID_OF_SAMPLE_IMAGES_FROM_EACH_CLASS
# ==============================================================================
def sample_images_from_each_class(X, Y, n_classes=25, n=5):
    """ Given a batch of images (stored as a numpy array), and an array of
        labels, It creates a grid of images such that:
            - Each row contains images for each class.
            - Each column contains the first `n` images for that class.

    Args:
        X: (numpy array)
            Array containing batch of images. Shape should be:
                - [n_batch, im_rows, im_cols, n_channels]
        Y: (list or numpy array)
            The class labels for each sample.
        n_classes: (int, or None)(default=25)
            The number of classes in the data.
            If this value is `None`, then the number of classes will be
            infered from the max val from Y.
        n:  (int)(default=5)
            The number of images to sample for each class.

    Returns: (numpy array)
        The grid of images as one large image of shape:
            - [n_classes*im_cols, num_per_class*im_rows, n_channels]
    """
    # TODO: maybe add random sampling.
    # TODO: Handle greyscale images with no channels dimension

    cols = n
    _, img_height, img_width, n_channels =  X.shape

    # Infer the number of classes if not provided
    if n_classes is None:
        n_classes = Y.max()

    # Initialize the grid
    grid_shape = (0, img_width * (cols+1), n_channels)
    grid = np.zeros(grid_shape, dtype=np.uint8)

    # FOR EACH CLASS
    for id in range(n_classes):
        # Extract the images for the current class id
        imgs = X[Y==id][:cols]
        row = batch_of_images_to_grid(imgs, rows=1, cols=cols)

        # Extract the label image - ignoring alpha channel
        img_file = os.path.join("images", "{:02d}.png".format(id))
        label_img = scipy.misc.imread(img_file)[:,:,:3] # Ignore the alpha chanel
        label_img = scipy.misc.imresize(label_img, [img_height, img_width]) # resize

        # Append row of samples to the grid
        row2 = np.append(label_img, row, axis=1) # Append label image and samples
        grid = np.append(grid, row2, axis=0)     # Append the row to the grid

    return grid



# ==============================================================================
#                                                                   MPL_SHOW_IMG
# ==============================================================================
def mpl_show_img(a, figsize=(15,10)):
    """Given a numpy array representing an image, view it (using matplotlib)"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.imshow(a,  cmap="gray")     # Can actually render RGB with "gray"
    ax.grid(False)                     # Remove gridline
    ax.get_yaxis().set_visible(False)  # Remove axis ticks
    ax.get_xaxis().set_visible(False)  # Remove axis ticks
    plt.show()

# ==============================================================================
#                                                                   TRAIN_CURVES
# ==============================================================================
def train_curves(train, valid, saveto=None, label="Accuracy over time"):
    """ Plots the training curves. If `saveto` is specified, it saves the
        the plot image to a file.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("Accuracies over time", fontsize=15)
    ax.plot(train, color="#FF4F40",  label="train")
    ax.plot(valid, color="#307EC7",  label="eval")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower right", title="", frameon=False,  fontsize=8)
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)


# ==============================================================================
#                                                         PLOT_LABEL_FREQUENCIES
# ==============================================================================
def plot_label_frequencies(y, dataname="", logscale=False, saveto=None, ratio=False):
    """ Plots the frequency of each label in the dataset."""
    vals, freqs = np.array(np.unique(y, return_counts=True))
    if ratio:
        freqs = freqs / float(len(y))

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("Distribution of Labels in {} dataset".format(dataname), fontsize=15)
    ax.bar(vals, freqs, alpha=0.5, color="#307EC7", edgecolor="b", align='center', width=0.8, lw=1)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Frequency")
    if logscale:
        ax.set_yscale('log')
    if saveto is not None:
        fig.savefig(saveto)
    else:
        plt.show()

# ==============================================================================
#                                                      PLOT_DENSITY_DISTRIBUTION
# ==============================================================================
def plot_density_distribution(x, saveto=None, logscale=False, dataname=""):
    """Plots a density distribution for visualizing how values are spread out"""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_style('whitegrid')
    sns.kdeplot(x.flatten(), bw=0.5, ax=ax)

    fig.suptitle("Density disribution of {} dataset".format(dataname), fontsize=15)
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequency")
    if logscale:
        ax.set_yscale('log')
    if saveto is not None:
        fig.savefig(saveto)
    else:
        plt.show()
