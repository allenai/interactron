import math
import logging

import torchvision.ops
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
from datetime import datetime
import json
import yaml
from PIL import ImageDraw, ImageFont

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from utils.constants import THOR_CLASS_IDS, tlvis_classes
from utils.detection_utils import iou, compute_AP, match_predictions_to_detections
from utils.model_utils import merge_batch_seq, unmerge_batch_seq
from utils.viz_utils import draw_pr_curve, draw_prediction_distribuion, draw_box
from utils.detection_utils import Images, Labels
from utils.storage_utils import collate_fn
from utils.transform_utis import transform, inv_transform
from models.detr_models.util.box_ops import box_cxcywh_to_xyxy
from datasets.sequence_dataset import SequenceDataset


class RandomPolicyEvaluator:

    def __init__(self, model, config, load_checkpoint=False):
        self.model = model
        if load_checkpoint:
            self.model.load_state_dict(
                torch.load(config.EVALUATOR.CHECKPOINT, map_location=torch.device('cpu'))['model'], strict=False)
        self.test_dataset = SequenceDataset(config.DATASET.TEST.IMAGE_ROOT, config.DATASET.TEST.ANNOTATION_ROOT,
                                        config.DATASET.TEST.MODE, transform=transform)
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model.to(self.device)

        self.out_dir = config.EVALUATOR.OUTPUT_DIRECTORY + "/" + datetime.now().strftime("%m-%d-%Y-%H:%M:%S") + "/"

    def evaluate(self, save_results=False):

        # prepare data folder if we are saving
        if save_results:
            os.makedirs(self.out_dir + "images/", exist_ok=True)

        model, config = self.model, self.config.EVALUATOR
        # model.train(False)
        loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                            collate_fn=collate_fn)

        detections = []
        for data in loader:

            # place data on the correct device
            data["frames"] = data["frames"].to(self.device)
            data["masks"] = data["masks"].to(self.device)
            data["category_ids"] = [[j.to(self.device) for j in i] for i in data["category_ids"]]
            data["boxes"] = [[j.to(self.device) for j in i] for i in data["boxes"]]

            # forward the model
            model.eval()
            predictions, losses = model(data)

            with torch.no_grad():
                for b in range(predictions["pred_boxes"].shape[0]):
                    img_detections = []
                    # get predictions and labels for this image
                    pred_boxes = box_cxcywh_to_xyxy(predictions["pred_boxes"][b][0])
                    pred_scores, pred_cats = predictions["pred_logits"][b][0].softmax(dim=-1).max(dim=-1)
                    gt_boxes = box_cxcywh_to_xyxy(data["boxes"][b][0])
                    gt_cats = data["category_ids"][b][0]
                    # remove background predictions
                    non_background_idx = pred_cats != 1235
                    pred_cats -= 1
                    pred_boxes = pred_boxes[non_background_idx]
                    pred_cats = pred_cats[non_background_idx]
                    pred_scores = pred_scores[non_background_idx]
                    # perform nms
                    pruned_idxs = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.5)
                    # print("non background:", torch.count_nonzero(non_background_idx).item(),
                    #       "nms 0.5:", pruned_idxs.shape[0],
                    #       "nms 0.25:", torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.25).shape[0],
                    #       "nms 0.75:", torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.75).shape[0]
                    #       )
                    # pruned_idxs = pruned_idxs[pred_cats[pruned_idxs] != 1235]
                    pred_cats = pred_cats[pruned_idxs]
                    pred_boxes = pred_boxes[pruned_idxs]
                    pred_scores = pred_scores[pruned_idxs]
                    # get sets of categories of predictions and labels
                    pred_cat_set = set(pred_cats)
                    gt_cat_set = set(gt_cats)
                    pred_only_cat_set = set(THOR_CLASS_IDS).intersection(pred_cat_set - gt_cat_set)
                    # add each prediction to the list of detections
                    for cat in gt_cat_set:
                        if torch.any(pred_cats == cat):
                            cat_pred_boxes = pred_boxes[pred_cats == cat]
                            cat_pred_scores = pred_scores[pred_cats == cat]
                            cat_gt_boxes = gt_boxes[gt_cats == cat]
                            cat_ious = torchvision.ops.box_iou(cat_pred_boxes, cat_gt_boxes)
                            cat_best_ious, cat_best_match_idx = match_predictions_to_detections(cat_ious)
                            for i in range(cat_ious.shape[0]):
                                if torch.any(cat_best_match_idx == i):
                                    img_detections.append({
                                        "iou": cat_ious[i].max().item(),
                                        "category_match": True,
                                        "type": "tp",
                                        "pred_cat": cat.item(),
                                        "pred_score": cat_pred_scores[i].item(),
                                        "box": [coord.item() for coord in cat_pred_boxes[i]],
                                        "area": ((cat_pred_boxes[i][2] - cat_pred_boxes[i][0])  *
                                                 (cat_pred_boxes[i][3] - cat_pred_boxes[i][1])).item(),
                                        "img": data["initial_image_path"][b]
                                    })
                                else:
                                    img_detections.append({
                                        "iou": cat_ious[i].max().item(),
                                        "category_match": True,
                                        "type": "fp",
                                        "pred_cat": cat.item(),
                                        "pred_score": cat_pred_scores[i].item(),
                                        "box": [coord.item() for coord in cat_pred_boxes[i]],
                                        "area": ((cat_pred_boxes[i][2] - cat_pred_boxes[i][0])  *
                                                 (cat_pred_boxes[i][3] - cat_pred_boxes[i][1])).item(),
                                        "img": data["initial_image_path"][b]
                                    })
                            for j in range(cat_ious.shape[1]):
                                if cat_best_ious[j] == 0.0:
                                    img_detections.append({
                                        "iou": 0.0,
                                        "category_match": False,
                                        "type": "fn",
                                        "pred_cat": cat.item(),
                                        "pred_score": 0.0,
                                        "box": [coord.item() for coord in cat_gt_boxes[j]],
                                        "area": ((cat_gt_boxes[j][2] - cat_gt_boxes[j][0])  *
                                                 (cat_gt_boxes[j][3] - cat_gt_boxes[j][1])).item(),
                                        "img": data["initial_image_path"][b]
                                    })
                        else:
                            cat_gt_boxes = gt_boxes[gt_cats == cat]
                            for j in range(cat_gt_boxes.shape[0]):
                                img_detections.append({
                                    "iou": 0.0,
                                    "category_match": False,
                                    "type": "fn",
                                    "pred_cat": cat.item(),
                                    "pred_score": 0.0,
                                    "box": [coord.item() for coord in cat_gt_boxes[j]],
                                    "area": ((cat_gt_boxes[j][2] - cat_gt_boxes[j][0])  *
                                                 (cat_gt_boxes[j][3] - cat_gt_boxes[j][1])).item(),
                                    "img": data["initial_image_path"][b]
                                })
                    for cat in pred_only_cat_set:
                        cat_pred_boxes = pred_boxes[pred_cats == cat]
                        cat_pred_scores = pred_scores[pred_cats == cat]
                        for i in range(cat_pred_scores.shape[0]):
                            img_detections.append({
                                "iou": 0.0,
                                "category_match": False,
                                "type": "fp",
                                "pred_cat": cat.item(),
                                "pred_score": cat_pred_scores[i].item(),
                                "box": [coord.item() for coord in cat_pred_boxes[i]],
                                "area": ((cat_pred_boxes[i][2] - cat_pred_boxes[i][0])  *
                                                 (cat_pred_boxes[i][3] - cat_pred_boxes[i][1])).item(),
                                "img": data["initial_image_path"][b],
                            })
                    detections = detections + img_detections
                    if save_results:
                        img = inv_transform(data["frames"][b][0].detach().cpu()).resize((1200, 1200))
                        # font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-C.ttf", 20)
                        font = ImageFont.load_default()
                        draw = ImageDraw.Draw(img)
                        for det in img_detections:
                            color = None
                            if det["type"] == "tp":
                                # color = "blue"
                                if det["iou"] >= 0.5:
                                    color = "blue"
                                else:
                                    color = "black"
                            if det["type"] == "fn":
                                continue
                            if det["type"] == "fp": # and det["pred_score"] > 0.5:
                                continue
                            if color is not None:
                                draw.rectangle([1200 * c for c in det["box"]], outline=color, width=2)
                                text = tlvis_classes[det["pred_cat"]]
                                x, y = 1200 * det["box"][0], 1200 * (det["box"][1] - 0.02)
                                w, h = font.getsize(text)
                                draw.rectangle((x, y, x + w, y + h), fill=color)
                                draw.text((x, y), text, fill="white", font=font)
                        img_root = self.out_dir + "images/"
                        img.save(img_root + img_detections[0]["img"].split("/")[-1])

        tps = [x for x in detections if x["type"] == "tp"]
        fps = [x for x in detections if x["type"] == "fp"]
        fns = [x for x in detections if x["type"] == "fn"]

        if not save_results:
            p, r, = self.compute_pr(detections, nsamples=100, iou_thresh=0.5)
            ap_50 = compute_AP([{"precision": p[i], "recall": r[i]} for i in range(len(p))])
            return ap_50, len(tps), len(fps), len(fns)

        aps = []
        for thresh in np.arange(0.5, 1.0, 0.05):
            p, r, = self.compute_pr(detections, nsamples=100, iou_thresh=thresh)
            aps.append(compute_AP([{"precision": p[i], "recall": r[i]} for i in range(len(p))]))

        aps_small = []
        for thresh in np.arange(0.5, 1.0, 0.05):
            p, r, = self.compute_pr(detections, nsamples=100, iou_thresh=thresh, min_area=0.0, max_area=32**2/300**2)
            aps_small.append(compute_AP([{"precision": p[i], "recall": r[i]} for i in range(len(p))]))

        aps_medium = []
        for thresh in np.arange(0.5, 1.0, 0.05):
            p, r, = self.compute_pr(detections, nsamples=100, iou_thresh=thresh,
                                    min_area=32**2/300**2, max_area=96**2/300**2)
            aps_medium.append(compute_AP([{"precision": p[i], "recall": r[i]} for i in range(len(p))]))

        aps_large = []
        for thresh in np.arange(0.5, 1.0, 0.05):
            p, r, = self.compute_pr(detections, nsamples=100, iou_thresh=thresh, min_area=96**2/300**2, max_area=1.0)
            aps_large.append(compute_AP([{"precision": p[i], "recall": r[i]} for i in range(len(p))]))

        p, r, = self.compute_pr(detections, nsamples=100, iou_thresh=0.5)
        ap_50 = compute_AP([{"precision": p[i], "recall": r[i]} for i in range(len(p))])
        print("AP_50:", ap_50)

        plt.plot(r, p)
        plt.title("PR Curve | AP_50=" + str(ap_50))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(self.out_dir + "pr_curve.png")

        results = {
            "AP_50": ap_50,
            "detections": detections
        }

        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.out_dir + "results.json", 'w') as f:
            json.dump(results, f)

    @staticmethod
    def compute_pr(detections, nsamples=100, iou_thresh=0.5, min_area=0.0, max_area=1.0):
        p = []
        r = []
        detections = [d for d in detections if min_area < d["area"] < max_area]
        tps = [d for d in detections if d["type"] == "tp"]
        fps = [d for d in detections if d["type"] == "fp"]
        fns = [d for d in detections if d["type"] == "fn"]
        i = 0
        while i < len(tps):
            if tps[i]["iou"] < iou_thresh:
                fps.append(tps.pop(i))
            else:
                i += 1
        for conf_thresh in np.arange(0.0, 1.0, 1.0/nsamples):
            i = 0
            while i < len(tps):
                if tps[i]["pred_score"] < conf_thresh:
                    tps.pop(i)
                else:
                    i += 1
            i = 0
            while i < len(fps):
                if fps[i]["pred_score"] < conf_thresh:
                    fps.pop(i)
                else:
                    i += 1
            p.append(0 if len(tps) == 0 else len(tps) / (len(tps) + len(fps)))
            r.append(0 if len(tps) == 0 else len(tps) / (len(tps) + len(fns)))

        return p, r

