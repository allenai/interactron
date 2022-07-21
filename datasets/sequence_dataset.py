import torch
from torch.utils.data import Dataset
import json
import random
from PIL import Image

from utils.constants import ACTIONS


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
        if self.mode == "test" and actions is None:
            actions = ['RotateLeft', 'MoveAhead', 'RotateLeft', 'MoveBack', 'RotateRight']

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
                state_name = state["actions"][actions[i]]
                state = scene["state_table"][state_name]

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
