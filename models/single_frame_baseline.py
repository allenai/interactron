import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detr import detr


class SingleFrameBaselineModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.detector = detr(config=cfg)
        self.cfg = cfg
        self.logger = None
        self.mode = 'train'

    def forward(self, data):
        predictions, losses = self.detector(data)
        predictions.nms(k=50)

        return predictions, losses

    def configure_optimizer(self, train_config):
        # optim_groups = self.detector.get_optimizer_groups(train_config)
        optim_groups = [{"params": list(self.model.parameters()), "weight_decay": train_config.WEIGHT_DECAY}]
        assert train_config.OPTIM_TYPE in ["Adam", "AdamW", "SGD"], \
            "Invalid optimizer type {}. Please select Adam, AdamW or SGD"
        if train_config.OPTIM_TYPE == "AdamW":
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.LEARNING_RATE,
                                          betas=(train_config.BETA1, train_config.BETA2))
        elif train_config.OPTIM_TYPE == "Adam":
            optimizer = torch.optim.Adam(optim_groups, lr=train_config.LEARNING_RATE,
                                          betas=(train_config.BETA1, train_config.BETA2))
        else:
            optimizer = torch.optim.SGD(optim_groups, lr=train_config.LEARNING_RATE, momentum=train_config.MOMENTUM)
        return optimizer

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger
        self.detector.set_logger(logger)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        self.detector.train(mode=False)
