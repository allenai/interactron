import torch
import torch.nn as nn

from models.detr_models.detr import build
from models.detr_models.util.misc import NestedTensor


class detr(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.model, self.criterion, self.postprocessor = build(config)
        self.model.load_state_dict(torch.load(config.WEIGHTS, map_location=torch.device('cpu'))['model'])
        self.logger = None
        self.mode = 'train'

    def predict(self, data):
        # reformat img and mask data
        b, s, c, w, h = data["frames"].shape
        img = data["frames"].view(b*s, c, w, h)
        mask = data["masks"].view(b*s, w, h)
        # reformat labels
        labels = []
        for i in range(b):
            for j in range(s):
                labels.append({
                    "labels": data["category_ids"][i][j],
                    "boxes": data["boxes"][i][j]
                })
        # get predictions and losses
        out = self.model(NestedTensor(img, mask))
        # loss = self.criterion(out, labels)
        # clean up predictions
        for key, val in out.items():
            out[key] = val.view(b, s, *val.shape[1:])

        return out

    def forward(self, data):
        # reformat img and mask data
        b, s, c, w, h = data["frames"].shape
        img = data["frames"].view(b*s, c, w, h)
        mask = data["masks"].view(b*s, w, h)
        # reformat labels
        labels = []
        for i in range(b):
            for j in range(s):
                labels.append({
                    "labels": data["category_ids"][i][j],
                    "boxes": data["boxes"][i][j]
                })
        # get predictions and losses
        out = self.model(NestedTensor(img, mask))
        loss = self.criterion(out, labels)
        # clean up predictions
        for key, val in out.items():
            out[key] = val.view(b, s, *val.shape[1:])

        return out, loss

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        # only train proposal generator of detector
        # self.model.backbone.eval()
        self.model.train(mode)
        return self

    def get_optimizer_groups(self, train_config):
        optim_groups = [{
            "params": list(self.model.parameters()), "weight_decay": 0.0
        }]
        return optim_groups

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger
