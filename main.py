import logging
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_cie76

logging.basicConfig(format="", level=logging.DEBUG)
log = logging.getLogger(__name__)


def find_histogram(clt: KMeans) -> np.ndarray:
    """creates histogram with k clusters.

    Args:
        clt (KMeans): kmeans cluster object.

    Returns:
        np.ndarray: numpy array of histogram.
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def color_match(dominant: np.ndarray, threshold: int = 30) -> str:
    """
    matches the dominant color to RED, GREEN, BLUE and OTHER.

    Args:
        dominant (np.ndarray): RBG values of dominant color.
        threshold (int, optional): threshold for matching. Defaults to 30.

    Returns:
        str: matching color for dominant color.
    """
    output = "other"

    img_color = rgb2lab(np.uint8(np.asarray([[dominant]])))
    red = rgb2lab(np.uint8(np.asarray([[[255, 0, 0 + 9]]])))
    green = rgb2lab(np.uint8(np.asarray([[[0, 128, 0]]])))
    blue = rgb2lab(np.uint8(np.asarray([[[0, 0, 255]]])))

    if deltaE_cie76(red, img_color) < threshold:
        output = "red"
    elif deltaE_cie76(green, img_color) < threshold:
        output = "green"
    elif deltaE_cie76(blue, img_color) < threshold:
        output = "blue"
    else:
        pass
    return output


def dominant_classes(image_folder: os.path) -> None:
    """Creates folders and add the images to their respective class.

    Args:
        image_folder (path): path for images

    Returns:
        None
    """

    for my_file in os.listdir(image_folder):
        if not my_file.startswith("."):
            img = cv2.imread(os.path.join(image_folder, my_file))

            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img1 = img1.reshape(
                (img1.shape[0] * img1.shape[1], 3)
            )  # represent as row*column,channel number

            clt = KMeans(n_clusters=3)  # cluster number

            clt.fit(img1)

            hist = find_histogram(clt)  # histogram for clusters

            index = np.argmax(hist)  # find most frequent

            dominant_color = clt.cluster_centers_[index]

            log.debug(f"finding dominant color for {my_file}")

            color = color_match(dominant_color)

            if not os.path.exists(cwd + "/" + color + "/"):
                os.makedirs(cwd + "/" + color + "/")

            cv2.imwrite(os.path.join(cwd + "/" + color + "/", my_file), img)

            log.debug(f"saved {my_file} to {color} folder")

    return


if __name__ == "__main__":
    cwd = os.getcwd()
    dominant_classes(cwd + "/images")
