import torch
import torch.nn.functional as F
import torchvision
import numpy as np


class Images:
    def __init__(self, batch_size, seq_len, device, images=None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.images = images

    def get_index(self, idx):
        index_images = Images(
            batch_size=1,
            seq_len=self.seq_len,
            device=self.device,
            images=self.images[idx:idx+1]
        )
        return index_images

    def get_seq(self, idx, end=None):
        if end is None:
            end = idx + 1
        index_images = Images(
            batch_size=self.batch_size,
            seq_len=end-idx,
            device=self.device,
            images=self.images[:, idx:end]
        )
        return index_images

    def get_images(self, flat=False):
        assert self.images is not None, "No images given to this Images object"
        if flat:
            return self.images.view(self.batch_size * self.seq_len, *self.images.shape[2:])
        return self.images

    def set_images(self, images, flat=False):
        assert (not flat and len(images.shape) == 5) or (flat and len(images.shape) == 4), \
            "images must be a 5D tensor if flat is false, or 4D tensor if flat is true"
        if images.device != self.device:
            images = images.to(self.device)
        if flat:
            self.images = images.view(self.batch_size, self.seq_len, -1, *images.shape[2:])
        else:
            self.images = images

    def to(self, device):
        self.device = device
        if self.seq_len is not None:
            self.seq_len = self.seq_len.to(device)
        if self.images is not None:
            self.images = self.images.to(device)


class Prediction:
    def __init__(self, batch_size, seq_len, device,
                 logits=None, boxes=None, box_features=None, image_features=None, logger=None, mode='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.logits = logits
        self.boxes = boxes
        self.box_features = box_features
        self.image_features = image_features
        self.logger = logger
        self.mode = mode

    def get_seq(self, idx, end=None):
        if end is None:
            end = idx + 1
        index_predictions = Prediction(
            batch_size=self.batch_size,
            seq_len=end-idx,
            device=self.device,
            boxes=self.boxes[:, idx:end],
            logits=self.logits[:, idx:end],
            box_features=self.box_features[:, idx:end],
            image_features=self.image_features[:, idx:end],
            logger=self.logger,
            mode=self.mode
        )
        return index_predictions

    def make_labels_from_predictions(self, c=0.5):
        self.nms(k=50)
        mask = (self.get_scores() > 0.5).view(-1)
        cats = self.get_categories().view(-1)
        boxes = self.get_boxes().view(-1, 4)
        label_cats = torch.ones_like(cats) * -1
        label_boxes = torch.ones_like(boxes) * -1.0
        label_cats[mask] = cats[mask]
        label_boxes[mask] = boxes[mask]
        return Labels(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            device=self.device,
            boxes=label_boxes.view(self.batch_size, self.seq_len, -1, 4),
            categories=label_cats.view(self.batch_size, self.seq_len, -1)
        )


    def get_logits(self, flat=False):
        assert self.logits is not None, "No logits given to this prediction object"
        if flat:
            return self.logits.view(self.batch_size * self.seq_len, *self.logits.shape[2:])
        return self.logits

    def set_logits(self, logits, flat=False):
        assert (not flat and len(logits.shape) == 4) or (flat and len(logits.shape) == 3), \
            "logits must be a 4D tensor if flat is false, or 3D tensor if flat is true"
        if logits.device != self.device:
            logits = logits.to(self.device)
        if flat:
            self.logits = logits.view(self.batch_size, self.seq_len, -1, *logits.shape[2:])
        else:
            self.logits = logits

    def get_categories(self, flat=False):
        assert self.logits is not None, "No logits given to this prediction object"
        logits = self.get_logits(flat=flat)
        return logits.argmax(dim=-1)

    def get_scores(self, flat=False):
        assert self.logits is not None, "No logits given to this prediction object"
        logits = self.get_logits(flat=flat)
        scores, _ = torch.max(F.softmax(logits, dim=-1)[:, :, :, :-1], dim=-1)
        return scores

    def get_boxes(self, flat=False):
        assert self.boxes is not None, "No boxes given to this prediction object, call select_boxes()"
        if flat:
            return self.boxes.view(self.batch_size * self.seq_len, *self.boxes.shape[2:])
        return self.boxes

    def set_boxes(self, boxes, flat=False):
        assert (not flat and len(boxes.shape) == 4) or (flat and len(boxes.shape) == 3), \
            "boxes must be a 4D tensor if flat is false, or 3D tensor if flat is true"
        if boxes.device != self.device:
            boxes = boxes.to(self.device)
        if flat:
            self.boxes = boxes.view(self.batch_size, self.seq_len, -1, *boxes.shape[2:])
        else:
            self.boxes = boxes

    def get_box_features(self, flat=False):
        assert self.box_features is not None, "No box features given to this prediction object"
        if flat:
            return self.box_features.view(self.batch_size * self.seq_len, *self.box_features.shape[2:])
        return self.box_features

    def set_box_features(self, box_features, flat=False):
        assert (not flat and len(box_features.shape) == 4) or (flat and len(box_features.shape) == 3), \
            "box features must be a 4D tensor if flat is false, or 3D tensor if flat is true"
        if box_features.device != self.device:
            box_features = box_features.to(self.device)
        if flat:
            self.box_features = box_features.view(self.batch_size, self.seq_len, -1, *box_features.shape[2:])
        else:
            self.box_features = box_features

    def get_image_features(self, flat=False):
        assert self.image_features is not None, "No image features given to this prediction object"
        if flat:
            return self.image_features.view(self.batch_size * self.seq_len, *self.image_features.shape[2:])
        return self.image_features

    def set_image_features(self, image_features, flat=False):
        assert (not flat and len(image_features.shape) == 5) or (flat and len(image_features.shape) == 4), \
            "image features must be a 5D tensor if flat is false, or 4D tensor if flat is true"
        if image_features.device != self.device:
            image_features = image_features.to(self.device)
        if flat:
            self.image_features = image_features.view(self.batch_size, self.seq_len, -1, *image_features.shape[2:])
        else:
            self.image_features = image_features

    def nms(self, k=50):
        logits = self.get_logits(flat=True)
        boxes = self.get_boxes(flat=True)
        box_features = self.get_box_features(flat=True)
        pruned_logits = torch.zeros(logits.shape[0], k, logits.shape[2], device=logits.device)
        pruned_logits[:, :, -1] = 1.0
        pruned_boxes = torch.zeros(boxes.shape[0], k, boxes.shape[2], device=boxes.device)
        pruned_box_features = torch.zeros(box_features.shape[0], k, box_features.shape[2], device=box_features.device)
        for n in range(logits.shape[0]):
            cats = logits[n, :, :-1].argmax(dim=-1)
            scores, _ = torch.max(F.softmax(logits[n], dim=-1)[:, :-1], dim=-1)
            pruned_indexes = torchvision.ops.batched_nms(boxes[n], scores, cats, iou_threshold=0.5)[:k]
            t = pruned_indexes.shape[0]
            pruned_logits[n][:t] = logits[n][pruned_indexes]
            pruned_boxes[n][:t] = boxes[n][pruned_indexes]
            pruned_box_features[n][:t] = box_features[n][pruned_indexes]
        self.set_logits(pruned_logits, flat=True)
        self.set_boxes(pruned_boxes, flat=True)
        self.set_box_features(pruned_box_features, flat=True)

    def to(self, device):
        self.device = device
        if self.logits is not None:
            self.logits = self.logits.to(device)
        if self.boxes is not None:
            self.boxes = self.boxes.to(device)
        if self.box_features is not None:
            self.box_features = self.box_features.to(device)
        if self.image_features is not None:
            self.image_features = self.image_features.to(device)


class Labels:
    def __init__(self, batch_size, seq_len, device,
                 boxes=None, categories=None, matched_boxes=None, matched_categories=None, episode_ids=None,
                 logger=None, mode='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.boxes = boxes
        self.categories = categories
        self.episode_ids = episode_ids
        self.matched_boxes = matched_boxes
        self.matched_categories = matched_categories
        self.logger = logger
        self.mode = mode

    def get_index(self, idx):
        index_labels = Labels(
            batch_size=1,
            seq_len=self.seq_len,
            device=self.device,
            boxes=self.boxes[idx:idx+1],
            categories=self.categories[idx:idx+1],
            episode_ids=self.episode_ids[idx:idx+1],
            matched_boxes=None,
            matched_categories=None,
            logger=self.logger,
            mode=self.mode
        )
        return index_labels

    def get_seq(self, idx, end=None):
        if end is None:
            end = idx + 1
        index_labels = Labels(
            batch_size=self.batch_size,
            seq_len=end-idx,
            device=self.device,
            boxes=self.boxes[:, idx:end],
            categories=self.categories[:, idx:end],
            matched_boxes=None,
            matched_categories=None,
            logger=self.logger,
            mode=self.mode
        )
        return index_labels

    def get_boxes(self, flat=False):
        assert self.boxes is not None, "No boxes given to this labels object"
        if flat:
            return self.boxes.view(self.batch_size * self.seq_len, *self.boxes.shape[2:])
        return self.boxes

    def set_boxes(self, boxes, flat=False):
        assert (not flat and len(boxes.shape) == 4) or (flat and len(boxes.shape) == 3), \
            "boxes must be a 4D tensor if flat is false, or 3D tensor if flat is true"
        if boxes.device != self.device:
            boxes = boxes.to(self.device)
        if flat:
            self.boxes = boxes.view(self.batch_size, self.seq_len, -1, *boxes.shape[2:])
        else:
            self.boxes = boxes

    def get_matched_boxes(self, flat=False):
        assert self.matched_boxes is not None, "No matched_boxes given to this labels object"
        if flat:
            return self.matched_boxes.view(self.batch_size * self.seq_len, *self.matched_boxes.shape[2:])
        return self.matched_boxes

    def set_matched_boxes(self, matched_boxes, flat=False):
        assert (not flat and len(matched_boxes.shape) == 4) or (flat and len(matched_boxes.shape) == 3), \
            "boxes must be a 4D tensor if flat is false, or 3D tensor if flat is true"
        if matched_boxes.device != self.device:
            matched_boxes = matched_boxes.to(self.device)
        if flat:
            self.matched_boxes = matched_boxes.view(self.batch_size, self.seq_len, -1, *matched_boxes.shape[2:])
        else:
            self.matched_boxes = matched_boxes

    def get_categories(self, flat=False):
        assert self.categories is not None, "No categories given to this labels object"
        if flat:
            return self.categories.view(self.batch_size * self.seq_len, *self.categories.shape[2:])
        return self.categories

    def set_categories(self, categories, flat=False):
        assert (not flat and len(categories.shape) == 3) or (flat and len(categories.shape) == 2), \
            "categories must be a 3D tensor if flat is false, or 2D tensor if flat is true"
        if categories.device != self.device:
            categories = categories.to(self.device)
        if flat:
            self.categories = categories.view(self.batch_size, self.seq_len, *categories.shape[1:])
        else:
            self.categories = categories

    def get_episode_ids(self, flat=False):
        assert self.episode_ids is not None, "No episode ids given to this labels object"
        if flat:
            return self.episode_ids.view(self.batch_size * self.seq_len, *self.episode_ids.shape[2:])
        return self.episode_ids

    def set_episode_ids(self, episode_ids, flat=False):
        assert (not flat and len(episode_ids.shape) == 3) or (flat and len(episode_ids.shape) == 2), \
            "episode_ids must be a 3D tensor if flat is false, or 2D tensor if flat is true"
        if episode_ids.device != self.device:
            episode_ids = episode_ids.to(self.device)
        if flat:
            self.categories = episode_ids.view(self.batch_size, self.seq_len, *episode_ids.shape[1:])
        else:
            self.categories = episode_ids

    def get_matched_categories(self, flat=False):
        assert self.matched_categories is not None, "No matched_categories given to this labels object"
        if flat:
            return self.matched_categories.view(self.batch_size * self.seq_len, *self.matched_categories.shape[2:])
        return self.matched_categories

    def set_matched_categories(self, matched_categories, flat=False):
        assert (not flat and len(matched_categories.shape) == 3) or (flat and len(matched_categories.shape) == 2), \
            "matched_categories must be a 3D tensor if flat is false, or 2D tensor if flat is true"
        if matched_categories.device != self.device:
            matched_categories = matched_categories.to(self.device)
        if flat:
            self.matched_categories = matched_categories.view(
                self.batch_size, self.seq_len, *matched_categories.shape[1:])
        else:
            self.matched_categories = matched_categories

    def match_labels(self, predictions):
        proposals = predictions.get_boxes(flat=True)
        cats = torch.ones(*proposals.shape[:2], device=proposals.device, dtype=torch.long)
        cats *= predictions.get_logits().shape[-1] - 1
        boxes = torch.zeros_like(proposals)

        gt_boxes = self.get_boxes(flat=True)
        gt_categories = self.get_categories(flat=True)
        for n in range(proposals.shape[0]):
            ious = torchvision.ops.box_iou(proposals[n], gt_boxes[n])
            max_ious, max_iou_idxs = ious.max(dim=1)
            match_mask = max_ious > 0.5
            # if torch.count_nonzero(match_mask) == 0:
            #     print("Yeah!")
            best_iou_categories = gt_categories[n][max_iou_idxs]
            cats[n][match_mask] = best_iou_categories[match_mask]
            best_iou_boxes = gt_boxes[n][max_iou_idxs]
            boxes[n][match_mask] = best_iou_boxes[match_mask].float()
            # logging
            if self.logger:
                num_objects_in_frame = torch.count_nonzero(gt_categories[n] != -1)
                num_objects_with_matched_predictions = torch.unique(max_iou_idxs[match_mask]).shape[0]
                self.logger.add_value(
                    "{}/Number of Objects Per Frame".format(self.mode.capitalize()), num_objects_in_frame
                )
                self.logger.add_value(
                    "{}/Number of Matched Predictions Per Frame".format(self.mode.capitalize()),
                    num_objects_with_matched_predictions
                )
        self.set_matched_categories(cats, flat=True)
        self.set_matched_boxes(boxes, flat=True)

    def to(self, device):
        self.device = device
        if self.boxes is not None:
            self.boxes = self.boxes.to(device)
        if self.categories is not None:
            self.categories = self.categories.to(device)
        if self.matched_boxes is not None:
            self.matched_boxes = self.matched_boxes.to(device)
        if self.matched_categories is not None:
            self.matched_categories = self.matched_categories.to(device)


def prune_predictions(logits, boxes, box_features, backbone_boxes, tensors_to_prude=[], k=50):
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


def match_predictions_to_detections(ious):
    p_preferences = torch.argsort(ious, dim=1, descending=True)
    p_preference_idxs = torch.zeros((ious.shape[0],), dtype=torch.long, device=ious.device)
    free_ps = torch.ones((ious.shape[0],), device=ious.device).bool()
    tentative_matches = -torch.ones(ious.shape[1], dtype=torch.long, device=ious.device)
    for i in range(ious.shape[1]):
        proposals = p_preferences[torch.arange(0, p_preferences.shape[0]), p_preference_idxs]
        for j in range(ious.shape[1]):
            new_match = torch.argmax(ious[:, j] * (proposals == j))
            if tentative_matches[j] != -1 and tentative_matches[j] != new_match:
                free_ps[tentative_matches[j]] = True
            tentative_matches[j] = new_match
            free_ps[tentative_matches[j]] = False
        p_preference_idxs[free_ps] += 1
        if torch.count_nonzero(~free_ps) >= min(ious.shape[0], ious.shape[1]): # torch.all(~free_ps):
            break
    best_idxs = tentative_matches
    best_ious = torch.zeros(best_idxs.shape[0], device=ious.device)
    best_ious[best_idxs != -1] = ious[best_idxs[best_idxs != -1], best_idxs != -1]
    best_idxs[best_ious == 0.0] = -1
    return best_ious, best_idxs


def iou(b1, b2):
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    i = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
    u = a1 + a2 - i
    return i / u


def compute_AP(points):
    points.sort(key=lambda x: x["recall"])
    aps = [points[0]["precision"]]
    idx = 0
    for cutoff in np.linspace(0.1, 1.0, 10):
        while idx < len(points) and points[idx]["recall"] < cutoff:
            idx += 1
        if points[-1]["recall"] < cutoff:
            aps.append(0)
        elif idx == 0:
            aps.append(points[0]["precision"])
        else:
            aps.append(points[idx-1]["precision"])
    return np.mean(aps)


def compute_true_AP(points):
    points.sort(key=lambda x: x["recall"])
    rsums = [points[0]["recall"] * points[0]["precision"]]
    rsums += [
        (points[i]["recall"]-points[i-1]["recall"]) * ((points[i]["precision"] + points[i-1]["precision"])/2)
        for i in range(1, len(points))
    ]
    ap = sum(rsums)
    return ap
