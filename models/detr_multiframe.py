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
        img = data["frames"].view(b, s, c, w, h)
        mask = data["masks"].view(b, s, w, h)
        # reformat labels
        labels = []
        for i in range(b):
            labels.append([])
            for j in range(s):
                labels[i].append({
                    "labels": data["category_ids"][i][j],
                    "boxes": data["boxes"][i][j]
                })

        losses = []
        out_logits_list = []
        out_boxes_list = []

        for task in range(b):
            # get predictions and losses
            # with torch.no_grad():
            detr_out = self.detector(NestedTensor(img[task], mask[task]))
            # unfold images back into batch and sequences
            # for key in detr_out:
            #     detr_out[key] = detr_out[key].view(b, s, *detr_out[key].shape[1:])
            detr_out["embedded_memory_features"] = detr_out["embedded_memory_features"].unsqueeze(0)
            detr_out["box_features"] = detr_out["box_features"].unsqueeze(0)
            detr_out["pred_logits"] = detr_out["pred_logits"].unsqueeze(0)
            detr_out["pred_boxes"] = detr_out["pred_boxes"].unsqueeze(0)
            out = self.fusion(detr_out)
            # out["pred_boxes"] = detr_out["pred_boxes"]
            # del out['actions']
            # for key in out:
            #     out[key] = out[key].reshape(b * s, *out[key].shape[2:])
            # for key in detr_out:
            #     detr_out[key] = detr_out[key].reshape(b * s, *detr_out[key].shape[2:])

            loss = self.criterion(out, labels[task], background_c=0.1)
            total_loss = loss["loss_ce"] + 5 * loss["loss_giou"] + 2 * loss["loss_bbox"]
            total_loss.backward()
            losses.append({k: v.detach() for k, v in loss.items()})
            # clean up predictions
            # for key, val in out.items():
            #     out[key] = val.view(b, s, *val.shape[1:])

            out_logits_list.append(out["pred_logits"])
            out_boxes_list.append(out["pred_boxes"])

        predictions = {"pred_logits": torch.stack(out_logits_list, dim=0), "pred_boxes": torch.stack(out_boxes_list, dim=0)}
        losses = {k.replace("loss", "loss_detector"):
                                    torch.mean(torch.stack([x[k] for x in losses]))
                                for k, v in losses[0].items()}

        return predictions, losses

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        self.detector.train(False)
        self.detector.transformer.decoder.train(mode)
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

