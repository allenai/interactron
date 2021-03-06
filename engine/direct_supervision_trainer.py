"""
Direct Supervision Random Training Loop
The model is trained on random sequences of data.
"""

import math

from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader

from datasets.sequence_dataset import SequenceDataset
from utils.transform_utis import transform, train_transform
from utils.logging_utils import TBLogger
from utils.storage_utils import collate_fn


class DirectSupervisionTrainer:

    def __init__(self, model, config, evaluator=None):
        self.model = model
        self.config = config
        self.evaluator = evaluator

        # set up logging and saving
        self.out_dir = os.path.join(self.config.TRAINER.OUTPUT_DIRECTORY, datetime.now().strftime("%m-%d-%Y:%H:%M:%S"))
        self.logger = TBLogger(os.path.join(self.out_dir, "logs"))
        self.model.set_logger(self.logger)
        self.checkpoint_path = os.path.join(self.out_dir, "detector.pt")
        self.saved_checkpoints = None

        self.train_dataset = SequenceDataset(config.DATASET.TRAIN.IMAGE_ROOT, config.DATASET.TRAIN.ANNOTATION_ROOT,
                                        config.DATASET.TRAIN.MODE, transform=train_transform)
        self.test_dataset = SequenceDataset(config.DATASET.TEST.IMAGE_ROOT, config.DATASET.TEST.ANNOTATION_ROOT,
                                        config.DATASET.TEST.MODE, transform=transform)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def record_checkpoint(self, w=1.0):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_parameters = raw_model.state_dict()
        if self.saved_checkpoints is None:
            print("New Save", w)
            self.saved_checkpoints = {k: w * v for k, v in raw_parameters.items()}
        else:
            print("Add on save", w)
            for param_name, weight in raw_parameters.items():
                self.saved_checkpoints[param_name] += w * weight

    def save_checkpoint(self):
        if self.saved_checkpoints is None:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            raw_parameters = raw_model.state_dict()
        else:
            raw_parameters = self.saved_checkpoints
        torch.save({"model": raw_parameters}, self.checkpoint_path)

    def train(self):
        model, config = self.model, self.config.TRAINER
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.Adam(raw_model.get_optimizer_groups(config), lr=config.LEARNING_RATE)

        def run_epoch(split):
            is_train = split == 'train'
            loader = DataLoader(self.train_dataset if is_train else self.test_dataset, shuffle=is_train,
                                pin_memory=True, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                                collate_fn=collate_fn)

            loss_list = []
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, data in pbar:

                # place data on the correct device
                data["frames"] = data["frames"].to(self.device)
                data["masks"] = data["masks"].to(self.device)
                data["category_ids"] = [[j.to(self.device) for j in i] for i in data["category_ids"]]
                data["boxes"] = [[j.to(self.device) for j in i] for i in data["boxes"]]

                # forward the model
                predictions, losses = model(data)
                loss = losses["loss_detector_ce"] + 5 * losses["loss_detector_bbox"] \
                       + 2 * losses["loss_detector_giou"]

                # log the losses
                for name, loss_comp in losses.items():
                    self.logger.add_value("{}/{}".format("Train" if is_train else "Test", name), loss_comp.mean())
                self.logger.add_value("{}/Total Loss".format("Train" if is_train else "Test"), loss.mean())
                loss_list.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_NORM_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()

                    # decay the learning rate based on our progress
                    if config.LR_DECAY:
                        self.tokens += data["frames"].shape[0]
                        if self.tokens < config.WARMUP_TOKENS:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.WARMUP_TOKENS))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.WARMUP_TOKENS) / \
                                       float(max(1, config.FINAL_TOKENS - config.WARMUP_TOKENS))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.LEARNING_RATE * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.LEARNING_RATE

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch} iter {it}: train loss {float(np.mean(loss_list)):.5f}. lr {lr:e}"
                    )

            if not is_train:
                test_loss = float(np.mean(loss_list))
                return test_loss

        def run_evaluation():
            test_loss = run_epoch('test')
            mAP_50, mAP, tps, fps, fns = self.evaluator.evaluate(save_results=False)
            self.logger.add_value("Test/TP", tps)
            self.logger.add_value("Test/FP", fps)
            self.logger.add_value("Test/FN", fns)
            self.logger.add_value("Test/mAP_50", mAP_50)
            self.logger.add_value("Test/mAP", mAP)
            model.zero_grad()
            return mAP

        self.tokens = 0  # counter used for learning rate decay
        run_evaluation()
        self.logger.log_values()
        for epoch in range(1, config.MAX_EPOCHS):
            run_epoch('train')
            if epoch % 1 == 0 and self.test_dataset is not None and self.evaluator is not None:
                run_evaluation()
            self.logger.log_values()

            if self.test_dataset is not None and config.MAX_EPOCHS - epoch <= config.SAVE_WINDOW:
                self.record_checkpoint(w=1/config.SAVE_WINDOW)
        self.save_checkpoint()
