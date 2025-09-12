import torch


def normalise(images):
    """rescales the array to have only values in [0,1]"""
    if images.max() - images.min() == 0:
        return images
    images = (images - images.min())/(images.max() - images.min())

    return images


def tonumpy(array):
    if type(array) is torch.Tensor:
        return array.detach().cpu().numpy()
    else:
        return array
