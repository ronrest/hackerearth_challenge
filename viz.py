import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#                                          GRID_OF_SAMPLE_IMAGES_FROM_EACH_CLASS
# ==============================================================================
def sample_images_from_each_class(X, Y, num_per_class=5, seed=None):
    """
    Given a batch of images (stored as a numpy array), and an array of labels,
    It creates a grid of images (randomly sampled), such that:
        - Each row contains images for each class.
        - Each column contains `num_per_class` randomly sampled images for
          that particular class.
    Args:
        X: (numpy array)
            Shape should be either:
                - [n_batch, im_rows, im_cols]
                - [n_batch, im_rows, im_cols, n_channels]
        Y: (list or numpy array)
            The class labels for each sample.
        num_per_class:  (int)
            The number of images to sample for each class.
        param seed: (int)
            Set the random seed.
    Returns: (numpy array)
        The grid of images as one large image of either shape:
            - [n_classes*im_cols, num_per_class*im_rows]
            - [n_classes*im_cols, num_per_class*im_rows, n_channels]
    """
    # TODO: have a resize option to rescale the individual sample images
    # Set the random seed if needed
    assert len(X) == len(Y), "X, and Y should have same number of samples"
    if seed is not None:
        np.random.seed(seed=seed)

    # Dimensions of the grid.
    n_classes = max(Y)+1
    im_shape = X[0].shape
    im_width = im_shape[1]
    im_height = im_shape[0]
    if len(im_shape)>2:
        n_channels = im_shape[2]
        grid_shape = (im_height * n_classes, im_width * num_per_class, n_channels)
    else:
        grid_shape = (im_height * n_classes, im_width * num_per_class)

    # Initialise the grid array
    grid_array = np.zeros(grid_shape, dtype=X[0].dtype)

    # For each class, sample num_per_class images and place them in grid
    for class_i in range(n_classes):
        available_pool = np.argwhere(np.squeeze(Y) == class_i).flatten()
        if len(available_pool) > 0:
            sample_indices = np.random.choice(available_pool, size=min(num_per_class, len(available_pool)), replace=False)
        else:
            # No samples available for this class
            continue

        # Append to corresponding position on grid
        for j, sample_index in enumerate(sample_indices):
            # Skip column if there is not enough samples from for this class
            if j >= len(available_pool):
                break
            row = im_height*class_i
            col = im_width*j

            grid_array[row:row+im_height, col:col+im_width] = X[sample_index]

    return grid_array

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

