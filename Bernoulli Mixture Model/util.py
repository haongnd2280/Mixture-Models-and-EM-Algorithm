from typing import List
import matplotlib.pyplot as plt


def image_transform(X: List[float]):
    """Convert gray scale image to binary image.
    """

    return (X / 128).astype("int")


def show_images(imgs: List[float], num_rows, num_cols, scale=2, titles: List[int] = None):
    """Show (a list of) binary images.
    """

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img, cmap=plt.cm.gray)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles is not None:
            ax.set_title(titles[i])

    plt.show()





