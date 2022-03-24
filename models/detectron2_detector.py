import torch
import torch.nn as nn
import torch.nn.functional as F
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.structures import Boxes, Instances
import torchvision
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from utils.detection_utils import Prediction


class Detectron2Detector(nn.Module):

    def __init__(
        self,
        config,
        model_config="COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
    ):
        super().__init__()
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.DEVICE = 'cpu'
        # Load custom model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1235
        cfg.MODEL.WEIGHTS = config.DETECTOR.WEIGHTS
        # Extract model from detectron2 predictor
        cfg.INPUT.MIN_SIZE_TEST = config.TEST_RESOLUTION
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        model = DefaultPredictor(cfg).model
        self.model = model
        self.resolution = config.TEST_RESOLUTION
        self.logger = None
        self.mode = 'train'

    def forward(self, x):
        with torch.no_grad():
            batched_inputs = self.preprocess_image(x.get_images(flat=True))
            images = self.model.preprocess_image(batched_inputs)
            features = self.backbone_forward(images.tensor)
            proposals, _ = self.model.proposal_generator(images, features, None)

        predictions = Prediction(x.batch_size, x.seq_len, x.device, logger=self.logger, mode=self.mode)
        predictions.set_image_features(features['res5'], flat=True)
        predictions = self.roi_heads_forward(features, proposals, predictions)

        return predictions

    def backbone_forward(self, x):
        # alternative backbone forward function that works with DataParallel
        outputs = {}
        x = self.model.backbone.stem(x)
        x = self.model.backbone.res2(x)
        x = self.model.backbone.res3(x)
        x = self.model.backbone.res4(x)
        x = self.model.backbone.res5(x)
        outputs['res5'] = x
        return outputs

    def roi_heads_forward(self, features, proposals, predictions):
        pruned_proposals = self.prune_proposals(proposals)
        features = [features[f] for f in self.model.roi_heads.box_in_features]
        box_features = self.model.roi_heads.box_pooler(features, [x.proposal_boxes for x in pruned_proposals])
        box_features = self.model.roi_heads.box_head(box_features)
        logits, boxes = self.model.roi_heads.box_predictor(box_features)
        # select only the box of the chosen categories
        box_class_indices = logits[:, :-1].argmax(dim=-1)
        boxes = boxes.view(boxes.shape[0], -1, 4)
        boxes = boxes[torch.arange(boxes.shape[0]), box_class_indices]
        # reshape outputs
        b = len(pruned_proposals)
        n = logits.shape[0] // b
        logits = logits.view(b, n, -1)
        box_features = box_features.view(b, n, -1)
        boxes = boxes.view(b, n, -1)
        # combine box anchors and box offsets
        for n in range(len(pruned_proposals)):
            boxes[n] += pruned_proposals[n].proposal_boxes.tensor

        predictions.set_logits(logits, flat=True)
        predictions.set_boxes(boxes, flat=True)
        predictions.set_box_features(box_features, flat=True)

        return predictions

    def preprocess_image(self, image):
        height, width = image.shape[1:3]
        image = F.interpolate(image.permute(0, 3, 1, 2).float(), size=self.resolution, mode='bilinear')
        img = [{"image": x, "height": height, "width": width} for x in image]
        return img

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

    def prune_proposals(self, proposals, k=1000):
        pruned_proposals = []
        for n in range(len(proposals)):
            padded_logits = torch.zeros(k, device=proposals[n].objectness_logits.device)
            padded_logits[:proposals[n].objectness_logits.shape[0]] = proposals[n].objectness_logits
            padded_boxes = torch.zeros(k, 4, device=proposals[n].objectness_logits.device)
            padded_boxes[:proposals[n].proposal_boxes.tensor.shape[0]] = proposals[n].proposal_boxes.tensor
            padded_boxes = Boxes(padded_boxes)
            pruned_proposals.append(Instances(
                image_size=proposals[n].image_size,
                objectness_logits=padded_logits,
                proposal_boxes=padded_boxes
            ))
        return pruned_proposals

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        # only train proposal generator of detector
        self.model.backbone.eval()
        self.model.proposal_generator.eval()
        # self.model.backbone.train(mode)
        # self.model.proposal_generator.train(mode)
        self.model.roi_heads.train(mode)
        # self.model.roi_heads.eval()
        return self

    def get_optimizer_groups(self, train_config):
        optim_groups = [{
            "params": list(self.model.roi_heads.parameters()), "weight_decay": train_config.WEIGHT_DECAY
        }]
        # optim_groups = []
        return optim_groups

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger
