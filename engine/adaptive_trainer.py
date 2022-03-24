"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import copy
import math

from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from utils.logging_utils import TBLogger
from utils.detection_utils import Images, Labels
from utils.viz_utils import draw_preds_and_labels
from utils.meta_utils import get_parameters, clone_parameters, set_parameters, sgd_step
from utils.config_utils import build_model
from utils.time_utils import Timer


class AdaptiveTrainer:

    def __init__(self, model, train_dataset, test_dataset, config, evaluator=None):
        self.detector = model.detector
        self.learned_loss = model.learned_loss
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.evaluator = evaluator

        # set up logging and saving
        self.out_dir = os.path.join(config.TRAINER.OUTPUT_DIRECTORY, datetime.now().strftime("%m-%d-%Y:%H:%M:%S"))
        os.makedirs(self.out_dir, exist_ok=True)
        self.logger = TBLogger(os.path.join(self.out_dir, "logs"))
        self.detector.set_logger(self.logger)
        self.learned_loss.set_logger(self.logger)
        self.detector_checkpoint_path = os.path.join(self.out_dir, "detector.pt")
        self.learned_loss_checkpoint_path = os.path.join(self.out_dir, "learned_loss.pt")

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.detector.to(self.device)
            self.learned_loss.to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        torch.save(self.detector.state_dict(), self.detector_checkpoint_path)
        torch.save(self.learned_loss.state_dict(), self.learned_loss_checkpoint_path)

    def train(self):
        detector, learned_loss, config = self.detector, self.learned_loss, self.config
        detector.train()
        # detector_optimizer = detector.configure_optimizer(config.TRAINER)
        learned_loss_optimizer = learned_loss.configure_optimizer(config.TRAINER)

        theta = get_parameters(detector.model)
        detector_optimizer = torch.optim.SGD(theta, lr=1e-3)

        def run_epoch(split):
            is_train = split == 'train'
            detector.train(True)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.TRAINER.BATCH_SIZE,
                                num_workers=config.TRAINER.NUM_WORKERS)

            losses = []
            ll_losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))  # if is_train else tqdm(enumerate(loader))

            for it, data in pbar:

                # place data on the correct device
                frames = data["frames"].to(self.device)
                gt_cats = data["category_ids"].to(self.device)
                gt_boxes = data["bounding_boxes"].to(self.device)

                # package data
                b, s = frames.shape[:2]
                images = Images(b, s, frames.device)
                images.set_images(frames)
                labels = Labels(b, s, frames.device, logger=self.logger, mode='train')
                labels.set_categories(gt_cats)
                labels.set_boxes(gt_boxes)

                # zero out gradient
                learned_loss_optimizer.zero_grad()
                detector_optimizer.zero_grad()

                # theta = get_parameters(detector.model)

                # forward the model
                for task_id in range(b):

                    # get the images and labels for this task
                    task_images = images.get_index(task_id)
                    task_labels = labels.get_index(task_id)

                    # make new instance of detector
                    task_detector = detector
                    clone_parameters(task_detector.model, iter(theta))

                    # perform preprocessing on model
                    with torch.no_grad():
                        preprocessed_task_predictions = detector.preprocess(task_images)

                    for k in range(config.TRAINER.NUM_GRAD_UPDATES):
                        # get paramerters of adaptable model
                        theta_task = get_parameters(task_detector.model)

                        # pass data through model
                        task_predictions, model_losses = task_detector(
                            preprocessed_task_predictions.get_seq(0, end=5),
                            task_labels.get_seq(0, end=5),
                            use_predictions_as_labels=False
                        )

                        if config.TRAINER.EXPLICIT_GRADIENT_MATCH:
                            # take adaptive gradient step
                            ll = learned_loss(task_predictions, images)
                            ll_grad = torch.autograd.grad(
                                ll,
                                theta_task,
                                create_graph=True,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            grad = torch.autograd.grad(
                                model_losses["category_prediction_loss"],  # ll,
                                theta_task,
                                create_graph=True,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            ll_loss = torch.stack([F.mse_loss(pred, gt) for pred, gt in zip(ll_grad, grad)]).sum()
                            ll_loss.backward(retain_graph=True)
                            ll_losses.append(ll_loss.item())
                        else:
                            ll = learned_loss(task_predictions, images)
                            grad = torch.autograd.grad(
                                ll,
                                theta_task,
                                create_graph=True,
                                retain_graph=True,
                                allow_unused=True,
                            )

                        sgd_step(task_detector.model, iter(grad), lr=1e-3)

                    # pass images through fully adapted adapted model
                    # task_predictions, model_losses = task_detector(preprocessed_task_predictions, task_labels)
                    target_predictions = preprocessed_task_predictions.get_seq(0)
                    target_labels = task_labels.get_seq(0)
                    task_predictions, model_losses = task_detector(target_predictions, target_labels)

                    # compute losses
                    bounding_box_loss = model_losses["bounding_box_loss"].mean()
                    category_prediction_loss = model_losses["category_prediction_loss"].mean()
                    task_loss = bounding_box_loss + category_prediction_loss

                    # log the task loss
                    self.logger.add_value("Train/Bounding Box Loss", bounding_box_loss.item())
                    self.logger.add_value("Train/Category Prediction Loss", category_prediction_loss.item())
                    self.logger.add_value("Train/Task Loss", task_loss.item())
                    losses.append(task_loss.detach().item())

                    # re-set original theta parameters and perform backwards pass
                    # set_parameters(task_detector.model, iter(theta))
                    task_loss.backward(retain_graph=True)

                if is_train:

                    # update the parameters
                    torch.nn.utils.clip_grad_norm_(detector.parameters(), config.TRAINER.GRAD_NORM_CLIP)
                    torch.nn.utils.clip_grad_norm_(learned_loss.parameters(), config.TRAINER.GRAD_NORM_CLIP)
                    learned_loss_optimizer.step()
                    detector_optimizer.step()

                    # zero out gradient
                    learned_loss_optimizer.zero_grad()
                    detector_optimizer.zero_grad()

                    # log the meta losses
                    self.logger.add_value("Train/Meta Loss", float(np.nanmean(losses)))
                    self.logger.add_value("Train/LL Loss", float(np.nanmean(ll_losses)))
                    # if it == 0:
                    #     self.logger.add_image(
                    #         "Train/Labels and Predictions",
                    #         draw_preds_and_labels(images, predictions, labels)
                    #     )

                    # decay the learning rate based on our progress
                    if config.TRAINER.LR_DECAY:
                        self.tokens += np.prod(b * s * 100)
                        if self.tokens < config.TRAINER.WARMUP_TOKENS:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.TRAINER.WARMUP_TOKENS))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.TRAINER.WARMUP_TOKENS) / \
                                       float(max(1, config.TRAINER.FINAL_TOKENS - config.TRAINER.WARMUP_TOKENS))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.TRAINER.LEARNING_RATE * lr_mult
                        for param_group in learned_loss_optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.TRAINER.LEARNING_RATE

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch} iter {it}: train loss {float(np.nanmean(losses)):.5f}. lr {lr:e}"
                    )

                else:
                    learned_loss_optimizer.zero_grad()
                    detector_optimizer.zero_grad()
                    # report progress
                    pbar.set_description(
                        f"iter {it}: test loss {float(np.nanmean(losses)):.5f}."
                    )

            set_parameters(detector.model, iter(theta))

            if not is_train:
                test_loss = float(np.nanmean(losses))
                return test_loss

        def run_evaluation():
            test_loss = run_epoch('test')
            mAP, tps, fps, fns = self.evaluator.evaluate(save_results=False)
            self.logger.add_value("Test/mAP", mAP)
            self.logger.add_value("Test/TP", tps)
            self.logger.add_value("Test/FP", fps)
            self.logger.add_value("Test/FN", fns)
            self.logger.add_value("Test/Loss", test_loss)
            return test_loss
            # return mAP

        best_ap = 0.0
        self.tokens = 0  # counter used for learning rate decay
        mAP = run_evaluation()
        # mAP = 0
        self.logger.log_values()
        for epoch in range(1, config.TRAINER.MAX_EPOCHS):
            run_epoch('train')
            if epoch % 10 == 0 and self.test_dataset is not None and self.evaluator is not None:
                mAP = run_evaluation()
            self.logger.log_values()

            # supports early stopping based on the test loss, or just save always if no test set is provided
            if self.test_dataset is not None and self.evaluator is not None and mAP > best_ap:
                best_ap = mAP
                self.save_checkpoint()
