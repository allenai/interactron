import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import cv2

from models.detectron2_detector import Detectron2Detector
from models.detr import DETRDetector
from models.gpt import GPT
from models.components import LinearBlock
from utils.constants import tlvis_classes
from utils.model_utils import merge_batch_seq, unmerge_batch_seq
from utils.detection_utils import iou
from utils.time_utils import Timer
from utils.viz_utils import draw_box


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        og_shape = x.shape
        x = self.model(x.view(-1, og_shape[-1]))
        return x.view(*og_shape[:-1], -1)


class FiveFrameBaselineModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.detector = DETRDetector(config=cfg)
        self.model = GPT(cfg.TRANSFORMER)
        self.proposal_encoder = LinearBlock(2264, cfg.TRANSFORMER.EMBEDDING_DIM, bias=False)
        self.img_feature_encoder = LinearBlock(2048, cfg.TRANSFORMER.EMBEDDING_DIM, bias=False)
        self.box_decoder = nn.Linear(in_features=1024, out_features=4, bias=False)
        self.category_decoder = nn.Linear(in_features=1024, out_features=1236, bias=False)
        self.cfg = cfg
        self.is_train = True
        self.timer = Timer()
        self.logger = None
        self.mode = 'train'

    def forward(self, images, labels):
        predictions = self.detector(images)
        predictions.nms(k=50)

        seq = self.fold_sequence(predictions)
        pred_embs = self.model(seq)[:, :250]
        pred_embs = F.gelu(pred_embs)
        predictions.set_logits(self.category_decoder(pred_embs), flat=True)
        # anchor_boxes = predictions.get_boxes()
        # anchor_boxes = anchor_boxes.view(anchor_boxes.shape[0], -1, anchor_boxes.shape[-1])
        # predictions.set_boxes(anchor_boxes + self.box_decoder(pred_embs), flat=True)
        predictions.nms(k=50)

        with torch.no_grad():
            labels.match_labels(predictions)

        # compute losses
        bounding_box_loss, category_loss = self.compute_losses(predictions, labels)
        losses = {
            "category_prediction_loss": category_loss,
            "bounding_box_loss": bounding_box_loss
        }

        return predictions, losses

    def compute_losses(self, predictions, labels):
        gt_cats = labels.get_matched_categories(flat=True)
        gt_boxes = labels.get_matched_boxes(flat=True)
        pred_logits = predictions.get_logits(flat=True)
        pred_boxes = predictions.get_boxes(flat=True)
        mask = gt_cats.view(-1) != self.cfg.DETECTOR.NUM_CLASSES
        # debugging
        if self.logger:
            self.logger.add_value(
                "{}/Number of matched ground truths".format(self.mode.capitalize()),
                torch.count_nonzero(mask)
            )
            self.logger.add_value(
                "{}/Number of positive detections".format(self.mode.capitalize()),
                torch.count_nonzero(predictions.get_logits().argmax(-1) != self.cfg.DETECTOR.NUM_CLASSES)
            )
        bbox_loss = F.mse_loss(pred_boxes.view(-1, 4)[mask], gt_boxes.view(-1, 4)[mask])
        category_loss = F.cross_entropy(pred_logits.view(-1, pred_logits.shape[-1]), gt_cats.view(-1))
        return bbox_loss, category_loss

    def configure_optimizer(self, train_config):
        optim_groups = self.model.get_optimizer_groups(train_config) + self.detector.get_optimizer_groups(train_config)
        optim_groups.append({
            "params": list(self.proposal_encoder.parameters()), "weight_decay": train_config.WEIGHT_DECAY
        })
        optim_groups.append({
            "params": list(self.img_feature_encoder.parameters()), "weight_decay": train_config.WEIGHT_DECAY
        })
        optim_groups.append({
            "params": list(self.box_decoder.parameters()), "weight_decay": train_config.WEIGHT_DECAY
        })
        optim_groups.append({
            "params": list(self.category_decoder.parameters()), "weight_decay": train_config.WEIGHT_DECAY
        })
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
        self.model.train(mode)
        self.proposal_encoder.train(mode)
        self.img_feature_encoder.train(mode)
        self.category_decoder.train(mode)
        self.box_decoder.train(mode)
        self.detector.train(mode)
        self.is_train = mode
        return self

    def match_proposals_to_labels(self, proposals, bounding_boxes, categories):
        labels = torch.ones(proposals.shape[0], proposals.shape[1], device=categories.device, dtype=torch.long)
        labels *= self.cfg.DETECTOR.NUM_CLASSES
        boxes = torch.zeros_like(proposals)
        for n in range(proposals.shape[0]):
            ious = torchvision.ops.box_iou(proposals[n], bounding_boxes[n])
            max_ious, max_iou_idxs = ious.max(dim=1)
            best_iou_categories = categories[n][max_iou_idxs]
            match_mask = max_ious > 0.5
            labels[n][match_mask] = best_iou_categories[match_mask]
            best_iou_boxes = bounding_boxes[n][max_iou_idxs]
            boxes[n][match_mask] = best_iou_boxes[match_mask].float()
        return labels, boxes

    def prune_predictions(self, logits, boxes, box_features, backbone_boxes, k=50):
        pruned_logits = torch.zeros(logits.shape[0], k, logits.shape[2], device=logits.device)
        pruned_logits[:, :, -1] = 1.0
        pruned_boxes = torch.zeros(boxes.shape[0], k, boxes.shape[2], device=boxes.device)
        pruned_backbone_boxes = torch.zeros_like(pruned_boxes)
        pruned_box_features = torch.zeros(box_features.shape[0], k, box_features.shape[2], device=box_features.device)
        for n in range(logits.shape[0]):
            cats = logits[n, :, :-1].argmax(dim=-1)
            scores, _ = torch.max(F.softmax(logits[n], dim=-1)[:, :-1], dim=-1)
            pruned_indexes = torchvision.ops.batched_nms(boxes[n], scores, cats, iou_threshold=0.5)[:k]
            t = pruned_indexes.shape[0]
            pruned_logits[n][:t] = logits[n][pruned_indexes]
            pruned_boxes[n][:t] = boxes[n][pruned_indexes]
            pruned_box_features[n][:t] = box_features[n][pruned_indexes]
            pruned_backbone_boxes[n][:t] = backbone_boxes[n][pruned_indexes]
        return pruned_logits, pruned_boxes, pruned_box_features, pruned_backbone_boxes

    def fold_sequence(self, predictions):
        img_features = predictions.get_image_features()
        box_features = predictions.get_box_features()
        boxes = predictions.get_boxes()
        logits = predictions.get_logits()
        detections = torch.cat((box_features, boxes, logits), dim=-1)
        b, t = img_features.shape[:2]
        img_features = img_features.permute(0, 1, 3, 4, 2)
        seq_img_features = self.img_feature_encoder(img_features.reshape(b, -1, img_features.shape[-1]))
        det_image_features = self.proposal_encoder(detections.reshape(b, -1, detections.shape[-1]))
        return torch.cat((det_image_features, seq_img_features), dim=1)

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger
        self.detector.set_logger(logger)
