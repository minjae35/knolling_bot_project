"""Differentiable coordinates; lengths in metres, angles in radians."""
import torch


def to_meters(uv):
    return uv * uv.new_tensor([0.6, 0.4])


def to_normalized(xy):
    return xy / xy.new_tensor([0.6, 0.4])


def to_pixels(uv):
    return torch.stack((576 * uv[..., 0] - 0.5, 384 * (1 - uv[..., 1]) - 0.5), -1)


def angle_delta(predicted, truth):
    difference = 2 * (predicted - truth)
    return 0.5 * torch.atan2(torch.sin(difference), torch.cos(difference))


def wrap_pi(theta):
    return torch.remainder(theta + torch.pi / 2, torch.pi) - torch.pi / 2
