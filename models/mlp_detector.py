import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import cv2

from models.detectron2_detector import Detectron2Detector
from models.gpt import GPT
from models.components import LinearBlock
from utils.constants import tlvis_classes
from utils.model_utils import merge_batch_seq, unmerge_batch_seq
from utils.detection_utils import iou
from utils.time_utils import Timer
from utils.viz_utils import draw_box
from utils.detection_utils import Prediction


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias),
            # nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        og_shape = x.shape
        x = self.model(x.view(-1, og_shape[-1]))
        return x.view(*og_shape[:-1], -1)


class MLPDetector(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model = nn.Sequential(
            LinearBlock(2264, 1024, bias=False),
            LinearBlock(1024, 1024, bias=False),
            nn.Linear(1024, 1236, bias=False),
        )
        self.preprocessor = Detectron2Detector(config=cfg)
        self.preprocessor.eval()
        self.cfg = cfg
        self.is_train = True
        self.timer = Timer()
        self.logger = None
        self.mode = 'train'

    def preprocess(self, images):
        predictions = self.preprocessor(images)
        predictions.nms(k=50)
        return predictions

    def forward(self, predictions, labels, use_predictions_as_labels=False):

        image_features = predictions.get_image_features(flat=True).detach()
        box_features = predictions.get_box_features(flat=True).detach()
        logits = predictions.get_logits(flat=True).detach()
        boxes = predictions.get_boxes(flat=True).detach()
        new_logits = self.model(torch.cat((box_features, logits, boxes), dim=-1))
        refined_predictions = Prediction(
            batch_size=predictions.batch_size,
            seq_len=predictions.seq_len,
            device=predictions.device,
            logits=new_logits.unsqueeze(0),
            boxes=boxes.unsqueeze(0),
            box_features=box_features.unsqueeze(0),
            image_features=image_features.unsqueeze(0)
        )

        if use_predictions_as_labels:
            labels = predictions.make_labels_from_predictions(c=0.8)

        with torch.no_grad():
            labels.match_labels(refined_predictions)

        # compute losses
        bounding_box_loss, category_loss = self.compute_losses(refined_predictions, labels)
        losses = {
            "category_prediction_loss": category_loss,
            "bounding_box_loss": 0.0 * bounding_box_loss  # torch.tensor([0.0], device=category_loss.device)
        }

        return refined_predictions, losses

    def compute_losses(self, predictions, labels):
        gt_cats = labels.get_matched_categories(flat=True)
        gt_boxes = labels.get_matched_boxes(flat=True)
        pred_logits = predictions.get_logits(flat=True)
        pred_boxes = predictions.get_boxes(flat=True)
        box_mask = gt_cats.view(-1) != self.cfg.DETECTOR.NUM_CLASSES
        # cat_mask = box_mask.detach().clone()
        # _, pred_rankings_mask = torch.topk(
        #     pred_logits.softmax(dim=-1)[:, :, -1].view(-1), int(len(cat_mask)*0.04))
        # cat_mask[pred_rankings_mask] = 1.0
        # debugging
        if self.logger:
            self.logger.add_value(
                "{}/Number of matched ground truths".format(self.mode.capitalize()),
                torch.count_nonzero(box_mask)
            )
            self.logger.add_value(
                "{}/Number of positive detections".format(self.mode.capitalize()),
                torch.count_nonzero(predictions.get_logits().argmax(-1) != self.cfg.DETECTOR.NUM_CLASSES)
            )
        bbox_loss = F.l1_loss(pred_boxes.view(-1, 4)[box_mask], gt_boxes.view(-1, 4)[box_mask])
        # nan guard
        if bbox_loss.isnan():
            bbox_loss = torch.zeros_like(bbox_loss)
        weights = torch.ones(self.cfg.DETECTOR.NUM_CLASSES + 1, device=gt_cats.device)
        weights[-1] = 1
        category_loss = F.cross_entropy(
            pred_logits.view(-1, pred_logits.shape[-1]),  # [cat_mask],
            gt_cats.view(-1),  # [cat_mask],
            weight=weights
        )
        return bbox_loss, category_loss

    def configure_optimizer(self, train_config):
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

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        self.preprocessor.train(False)
        self.model.train(mode)
        self.is_train = mode
        return self

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger
