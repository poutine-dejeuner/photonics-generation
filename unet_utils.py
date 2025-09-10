"""
UNet-specific utilities to avoid circular imports.
"""
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange


def display_reverse(images: list, savepath: str, idx: int):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.savefig(os.path.join(savepath, f"im{idx}.png"))
    plt.close()


class UNetPad():
    """
    Pads a tensor x of shape (B, C, H, W) so that H and W are divisible by 2^depth.
    Returns the padded tensor and the slices to undo the padding.
    """

    def __init__(self, sample: torch.Tensor, depth: int):
        self.depth = depth
        _, _, h, w = sample.shape
        target_h = ((h - 1) // 2**depth + 1) * 2**depth
        target_w = ((w - 1) // 2**depth + 1) * 2**depth
        pad_h = target_h - h
        pad_w = target_w - w
        self.pad = (0, pad_w, 0, pad_h)  # pad W then H
        self.unpad_slices = [slice(0, h), slice(0, w)]

    def __call__(self, x):
        return F.pad(x, self.pad)

    def inverse(self, x_padded):
        return x_padded[..., *self.unpad_slices]


def pad_to_unet(x: torch.Tensor, depth: int = 4):
    """
    Pads a tensor x of shape (B, C, H, W) so that H and W are divisible by 2^depth.
    Returns the padded tensor and the slices to undo the padding.
    """
    _, _, h, w = x.shape
    target_h = ((h - 1) // 2**depth + 1) * 2**depth
    target_w = ((w - 1) // 2**depth + 1) * 2**depth
    pad_h = target_h - h
    pad_w = target_w - w
    pad = (0, pad_w, 0, pad_h)  # pad W then H
    x_padded = F.pad(x, pad)
    return x_padded, (slice(0, h), slice(0, w))  # for unpadding later


class unet_pad_fun():
    def __init__(self, num_layers, data_sample):
        self.N = 2**num_layers

        def difference_with_next_multiple(x):
            reste = x % self.N
            if reste == 0:
                return 0
            else:
                return self.N - reste

        # Get the last 2 dimensions (H, W)
        h, w = data_sample.shape[-2:]
        
        # Calculate padding needed for each dimension
        h_pad_needed = difference_with_next_multiple(h)
        w_pad_needed = difference_with_next_multiple(w)
        
        # Split padding symmetrically
        self.h_pad_left = h_pad_needed // 2
        self.h_pad_right = h_pad_needed - self.h_pad_left
        self.w_pad_left = w_pad_needed // 2
        self.w_pad_right = w_pad_needed - self.w_pad_left
        
        # Store for F.pad format: [w_left, w_right, h_left, h_right]
        self.padding = [self.w_pad_left, self.w_pad_right, self.h_pad_left, self.h_pad_right]

    def pad(self, x):
        # F.pad expects padding in format [w_left, w_right, h_left, h_right, ...]
        # For tensors with more than 2 spatial dims, pad with zeros for extra dims
        n_extra_dims = x.ndim - 2  # Number of non-spatial dimensions (batch, channel, etc.)
        full_padding = self.padding + [0, 0] * (n_extra_dims - 2) if n_extra_dims > 2 else self.padding
        
        padded_tensor = F.pad(x, full_padding, mode='constant', value=0)
        return padded_tensor

    def crop(self, x):
        # Create slices for cropping
        h_slice = slice(self.h_pad_left, -self.h_pad_right if self.h_pad_right > 0 else None)
        w_slice = slice(self.w_pad_left, -self.w_pad_right if self.w_pad_right > 0 else None)
        
        cropped_tensor = x[..., h_slice, w_slice]
        return cropped_tensor
