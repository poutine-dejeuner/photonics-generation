import os
import numpy as np

from photo_gen.evaluation.evaluation import plot_closest_images

def plot_closest_images_test():
    imagespath = "../../data/diffusion/images.npy"
    trainsetpath = "../../data/topoptim/images.npy"
    images = np.load(os.path.expanduser(imagespath))
    train_set = np.load(os.path.expanduser(trainsetpath))
    savepath = "./"
    plot_closest_images(images, train_set, savepath)

if __name__ == "__main__":
    plot_closest_images_test()