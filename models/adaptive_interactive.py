import torch
import torch.nn as nn
import copy

from models.learned_loss_old import LearnedLossModel
from models.single_frame_baseline import SingleFrameBaselineModel
from models.detr import DETRDetector
from utils.meta_utils import get_parameters, clone_parameters, set_parameters, sgd_step, detach_parameters
from utils.detection_utils import Prediction


class AdaptiveModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.detector = DETRDetector(cfg)
        self.learned_loss = LearnedLossModel(cfg)

    def forward(self, images, labels):
        detector, learned_loss = self.detector, self.learned_loss
        logits_list = []
        boxes_list = []
        box_features_list = []
        image_features_list = []
        losses = []

        theta = get_parameters(detector)

        for task_id in range(images.batch_size):
            # get the images and labels for this task
            task_images = images.get_index(task_id)
            task_labels = labels.get_index(task_id)

            # load new weights into the model
            task_detector = detector
            clone_parameters(task_detector, iter(theta))

            for k in range(1):
                # get paramerters of adaptable model
                theta_task = get_parameters(task_detector)

                # pass data through model
                task_predictions, model_losses = task_detector(
                    task_images.get_seq(0, end=5),
                    task_labels.get_seq(0, end=5),
                    use_predictions_as_labels=False
                )

                # take adaptive gradient step
                ll, _ = learned_loss(task_predictions, images)
                grad = torch.autograd.grad(
                    ll,
                    theta_task,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )
                sgd_step(task_detector, iter(grad), lr=1e-3)
                # clear out gradients between each step when doing evaluation
                ll.backward()
                learned_loss.zero_grad()
                detach_parameters(task_detector)

            with torch.no_grad():
                # pass images through fully adapted adapted model
                # task_predictions, model_losses = task_detector(preprocessed_task_predictions, task_labels)
                target_predictions = task_images.get_seq(0)
                target_labels = task_labels.get_seq(0)
                task_predictions, model_losses = task_detector(target_predictions, target_labels)

                logits_list.append(task_predictions.get_logits().detach())
                boxes_list.append(task_predictions.get_boxes().detach())
                box_features_list.append(task_predictions.get_box_features().detach())
                image_features_list.append(task_predictions.get_image_features().detach())
                losses.append({k: v.detach() for k, v in model_losses.items()})

        set_parameters(detector, iter(theta))


        total_prediction = Prediction(
            len(logits_list),
            task_predictions.seq_len,
            task_predictions.device,
            logits=torch.cat(logits_list, dim=0),
            boxes=torch.cat(boxes_list, dim=0),
            box_features=torch.cat(box_features_list, dim=0),
            image_features=torch.cat(image_features_list, dim=0),
            logger=task_predictions.logger,
            mode=task_predictions.mode
        )
        total_losses = {k: sum([x[k].detach().item() for x in losses]) for k in losses[0]}
        return total_prediction, total_losses

    def get_next_action(self, images, labels):

        # pass data through model
        task_predictions, model_losses = (
            images, labels
        )

        # compute actions
        _, actions = self.learned_loss(task_predictions, images)
        return actions[images.seq_len]

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        self.detector.train()
        self.learned_loss.train()
        return self
