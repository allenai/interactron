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


class LearnedLossModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
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
        if cfg.TRANSFORMER.PREDICT_ACTIONS:
            self.policy_tokens = nn.Parameter(1, 5, cfg.TRANSFORMER.EMBEDDING_DIM)

    def forward(self, predictions, images):

        seq = self.fold_sequence(predictions)
        if self.cfg.TRANSFORMER.PREDICT_ACTIONS:
            seq = torch.cat((seq, self.policy_tokens), dim=1)
        out = self.model(seq)
        pred_embs = out[:, :250]
        learned_loss = torch.norm(pred_embs, p=2)

        if self.cfg.TRANSFORMER.PREDICT_ACTIONS:
            action_predictions = out[:, -5:]
            return learned_loss, action_predictions

        return learned_loss

    def configure_optimizer(self, train_config):
        optim_groups = self.model.get_optimizer_groups(train_config)
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
        self.is_train = mode
        return self

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
