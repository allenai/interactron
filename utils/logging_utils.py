import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TBLogger:

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.scalar_buffer = {}
        self.img_buffer = {}
        self.iter_counter = 0

    def add_value(self, name, value):
        assert any([isinstance(value, t) for t in [int, float, np.ndarray, torch.Tensor]]), \
            "Invalid type {}. Only int, float, np.ndarray and torch.Tensor are accepted".format(type(value))
        if isinstance(value, torch.Tensor):
            assert len(value.shape) == 0, \
                "Got tensor of shape {}. Only single value tensors are valid.".format(value.shape)
            value = value.item()
        if name in self.scalar_buffer:
            self.scalar_buffer[name].append(value)
        else:
            self.scalar_buffer[name] = [value]

    def add_image(self, name, img):
        assert any([isinstance(img, t) for t in [torch.Tensor]]), \
            "Invalid type {}. Only torch.Tensor are accepted".format(type(img))
        self.img_buffer[name] = img

    def log_values(self):
        # log all scalars
        for name, values in self.scalar_buffer.items():
            self.writer.add_scalar(name, np.mean(values), self.iter_counter)
        # log all images
        for name, value in self.img_buffer.items():
            self.writer.add_image(name, value, self.iter_counter, dataformats='HWC')
        # clear out buffers
        self.scalar_buffer = {}
        self.img_buffer = {}
        self.iter_counter += 1
