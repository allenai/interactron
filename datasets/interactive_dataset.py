import torch
from torch.utils.data import Dataset
import json
import random
from PIL import Image

from utils.constants import ACTIONS


class InteractiveDaatset(Dataset):
    """Interactive Dataset."""

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
        # interactive
        self.idx = -1
        self.actions = []

    def reset(self):

        self.idx += 1
        if self.idx >= len(self.annotations["data"]):
            self.idx = 0
        self.actions = []
        scene = self.annotations["data"][self.idx]

        state_name = scene["root"]
        state = scene["state_table"][state_name]
        actions = self.actions
        frames = []
        masks = []
        object_ids = []
        category_ids = []
        bounding_boxes = []
        initial_img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
        for i in range(len(self.actions)+1):
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
            if i < len(actions):
                state_name = state["actions"][actions[i]]
                state = scene["state_table"][state_name]

        sample = {
            'frames': torch.stack(frames, dim=0).unsqueeze(0),
            "masks": torch.stack(masks, dim=0).unsqueeze(0),
            "actions": torch.tensor([ACTIONS.index(a) for a in actions], dtype=torch.long).unsqueeze(0),
            "category_ids": [category_ids],
            "boxes": [bounding_boxes],
            "episode_ids": self.idx,
            "initial_image_path": [initial_img_path]
        }

        return sample

    def step(self, action):

        self.actions.append(ACTIONS[action])
        scene = self.annotations["data"][self.idx]

        state_name = scene["root"]
        state = scene["state_table"][state_name]
        actions = self.actions
        frames = []
        masks = []
        object_ids = []
        category_ids = []
        bounding_boxes = []
        initial_img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
        for i in range(len(self.actions)+1):
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
            if i < len(actions):
                state_name = state["actions"][actions[i]]
                state = scene["state_table"][state_name]

        sample = {
            'frames': torch.stack(frames, dim=0).unsqueeze(0),
            "masks": torch.stack(masks, dim=0).unsqueeze(0),
            "actions": torch.tensor([ACTIONS.index(a) for a in actions], dtype=torch.long).unsqueeze(0),
            "object_ids": object_ids,
            "category_ids": [category_ids],
            "boxes": [bounding_boxes],
            "episode_ids": self.idx,
            "initial_image_path": [initial_img_path]
        }

        return sample

    def __len__(self):
        return len(self.annotations["data"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scene = self.annotations["data"][idx]

        state_name = scene["root"]
        state = scene["state_table"][state_name]
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
                img_class_ids.append(v["category_id"])
                # # convery bbox coordinates from xywh to cxcywh
                # w, h = v["bbox"][-2:]
                # cx = v["bbox"][0] + (w / 2)
                # cy = v["bbox"][1] + (h / 2)
                # # normalize between 0.0 and 1.0
                # img_bounding_boxes.append([cx / imgw, cy / imgh, w / imgw, h / imgh])
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
                if self.mode == "test":
                    state_name = state["actions"][actions[i]]
                else:
                    state_name = random.choice(list(scene["state_table"]))
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

