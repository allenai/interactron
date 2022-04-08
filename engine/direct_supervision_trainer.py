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

        self.train_dataset = SequenceDataset(config.DATASET.TRAIN.IMAGE_ROOT, config.DATASET.TRAIN.ANNOTATION_ROOT,
                                        config.DATASET.TRAIN.MODE, transform=train_transform)
        self.test_dataset = SequenceDataset(config.DATASET.TEST.IMAGE_ROOT, config.DATASET.TEST.ANNOTATION_ROOT,
                                        config.DATASET.TEST.MODE, transform=transform)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({"model": raw_model.state_dict()}, self.checkpoint_path)

    def train(self):
        model, config = self.model, self.config.TRAINER
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.AdamW(raw_model.get_optimizer_groups(config), lr=3e-4)
        def run_epoch(split):
            is_train = split == 'train'
            # model.train(False)
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

                # shuffle positions
                # order = torch.randperm(5)
                # data["frames"] = data["frames"][:, order]
                # data["masks"] = data["masks"][:, order]
                # data["category_ids"] = [[b[i.item()] for i in order] for b in data["category_ids"]]
                # data["boxes"] = [[b[i.item()] for i in order] for b in data["boxes"]]

                # forward the model
                predictions, losses = model(data)
                loss = losses["loss_ce"] + losses["loss_bbox"] + losses["loss_giou"]

                # log the losses
                for name, loss_comp in losses.items():
                    self.logger.add_value("{}/{}".format("Train" if is_train else "Test", name), loss_comp.mean())
                self.logger.add_value("{}/Total Loss".format("Train" if is_train else "Test"), loss.mean())
                loss_list.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_NORM_CLIP)
                    optimizer.step()

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
            self.logger.add_value("Test/mAP_50", mAP_50)
            self.logger.add_value("Test/mAP", mAP)
            return mAP

        best_ap = 0.0
        self.tokens = 0  # counter used for learning rate decay
        mAP = run_evaluation()
        self.logger.log_values()
        for epoch in range(1, config.MAX_EPOCHS):
            run_epoch('train')
            if epoch % 1 == 0 and self.test_dataset is not None and self.evaluator is not None:
                mAP = run_evaluation()
            self.logger.log_values()

            # supports early stopping based on the test loss, or just save always if no test set is provided
            if self.test_dataset is not None and self.evaluator is not None and mAP > best_ap:
                best_ap = mAP
                self.save_checkpoint()
