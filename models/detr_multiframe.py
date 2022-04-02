import torch
import torch.nn as nn

from models.detr_models.detr import build
from models.detr_models.util.misc import NestedTensor
from models.transformer import Transformer


class detr_multiframe(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        # build DETR detector
        self.detector, self.criterion, self.postprocessor = build(config)
        self.detector.load_state_dict(torch.load(config.WEIGHTS, map_location=torch.device('cpu'))['model'])
        # build fusion transformer
        self.fusion = Transformer(config)
        self.logger = None
        self.mode = 'train'

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
        with torch.no_grad():
            detr_out = self.detector(NestedTensor(img, mask))
        # unfold images back into batch and sequences
        for key in detr_out:
            detr_out[key] = detr_out[key].view(b, s, *detr_out[key].shape[1:])
        out = self.fusion(detr_out)
        out["pred_boxes"] = detr_out["pred_boxes"]
        del out['actions']
        for key in out:
            out[key] = out[key].reshape(b * s, *out[key].shape[2:])
        for key in detr_out:
            detr_out[key] = detr_out[key].reshape(b * s, *detr_out[key].shape[2:])

        loss = self.criterion(out, labels, detector_out=detr_out)
        # clean up predictions
        for key, val in out.items():
            out[key] = val.view(b, s, *val.shape[1:])

        return out, loss

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        self.detector.train(False)
        self.fusion.train(mode)
        return self

    def get_optimizer_groups(self, train_config):
        optim_groups = [
            {"params": list(self.detector.parameters()), "weight_decay": 0.0},
            {"params": list(self.fusion.parameters()), "weight_decay": 0.0},
        ]
        return optim_groups

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger

