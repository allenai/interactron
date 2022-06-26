import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import json
import random
import time
import cv2
import hashlib
import numpy as np
from PIL import Image

from utils.constants import ACTIONS
from models.detr_models.util.box_ops import box_xyxy_to_cxcywh


class SequenceDataset(Dataset):
    """Sequence Rollout Dataset."""

    def __init__(self, img_root, annotations_path, mode="train", transform=None):
        """
        Args:
            root_dir (string): Directory with the train and test images and annotations
            test: Flag to indicate if the train or test set is used
        """
        assert mode in ["train", "test"], "Only train and test modes supported"
        self.mode = mode
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        # remove trailing slash if present
        self.img_dir = img_root if img_root[-1] != "/" else img_root[:-1]
        self.transform = transform

    def __len__(self):
        return len(self.annotations["data"])

    def __getitem__(self, idx, actions=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scene = self.annotations["data"][idx]

        # seed the random generator
        if self.mode == "test":
            actions = ['MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead', 'MoveAhead']

        state_name = scene["root"]
        state = scene["state_table"][state_name]
        if actions is None:
            actions = [random.choice(self.annotations["metadata"]["actions"]) for _ in range(5)]
        frames = []
        masks = []
        object_ids = []
        category_ids = []
        bounding_boxes = []
        initial_img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
        for i in range(5):
            # load image
            img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
            frame = Image.open(img_path)
            # get img dimensions
            imgw, imgh = frame.size
            masks.append(torch.zeros((imgw, imgh), dtype=torch.long))
            img_object_ids = []
            img_class_ids = []
            img_bounding_boxes = []
            for k, v in state["detections"].items():
                img_object_ids.append(hash(k.encode()))
                img_class_ids.append(v["category_id"]+1)
                w, h, cw, ch = v["bbox"]
                img_bounding_boxes.append([w, h, w+cw, h+ch])
            # bounding_boxes.append(img_bounding_boxes)
            # apply transforms to image
            if len(img_bounding_boxes) != 0:
                boxes = torch.tensor(img_bounding_boxes, dtype=torch.float)
                targets = {
                    "boxes": boxes,
                    "labels": torch.tensor(img_class_ids, dtype=torch.long),
                    "areas": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                    "iscrowd": torch.zeros((len(img_object_ids))).bool()
                }
            else:
                targets = None
            if self.transform:
                frame, targets = self.transform(frame, targets)
            frames.append(frame)
            bounding_boxes.append(targets["boxes"] if targets is not None else torch.zeros(0, 4))
            object_ids.append(img_object_ids)
            category_ids.append(targets["labels"] if targets is not None else torch.zeros(0).long())
            if i < 4:
                # if self.mode == "test":
                #     state_name = state["actions"][actions[i]]
                # else:
                #     state_name = random.choice(list(scene["state_table"]))
                state_name = state["actions"][actions[i]]
                state = scene["state_table"][state_name]

        # print(actions)
        # import matplotlib.pyplot as plt
        # for i, frame in enumerate(frames):
        #     plt.subplot(1, 5, i+1)
        #     plt.imshow(frame.detach().cpu().permute(1, 2, 0).numpy())
        # plt.show()

        sample = {
            'frames': frames,
            "masks": masks,
            "actions": [ACTIONS.index(a) for a in actions],
            "object_ids": object_ids,
            "category_ids": category_ids,
            "boxes": bounding_boxes,
            "episode_ids": idx,
            "initial_image_path": initial_img_path
        }

        return sample
