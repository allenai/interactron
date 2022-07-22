import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models.detr_models.detr import build
from models.detr_models.util.misc import NestedTensor
from models.transformer import Transformer
from utils.meta_utils import get_parameters, clone_parameters, sgd_step, set_parameters, detach_parameters, \
    detach_gradients
from utils.storage_utils import PathStorage


class interactron(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        # build DETR detector
        self.detector, self.criterion, self.postprocessor = build(config)
        self.detector.load_state_dict(torch.load(config.WEIGHTS, map_location=torch.device('cpu'))['model'])
        # build fusion transformer
        self.fusion = Transformer(config)
        self.logger = None
        self.mode = 'train'
        self.path_storage = {}
        self.config = config

    def predict(self, data):

        # reformat img and mask data
        b, s, c, w, h = data["frames"].shape
        img = data["frames"].view(s, c, w, h)
        mask = data["masks"].view(s, w, h)

        theta = get_parameters(self.detector)
        theta_task = detach_parameters(clone_parameters(theta))

        # get supervisor grads
        set_parameters(self.detector, theta_task)
        pre_adaptive_out = self.detector(NestedTensor(img, mask))
        pre_adaptive_out["embedded_memory_features"] = pre_adaptive_out["embedded_memory_features"].unsqueeze(0)
        pre_adaptive_out["box_features"] = pre_adaptive_out["box_features"].unsqueeze(0)
        pre_adaptive_out["pred_logits"] = pre_adaptive_out["pred_logits"].unsqueeze(0)
        pre_adaptive_out["pred_boxes"] = pre_adaptive_out["pred_boxes"].unsqueeze(0)

        fusion_out = self.fusion(pre_adaptive_out)
        learned_loss = torch.norm(fusion_out["loss"])
        detector_grad = torch.autograd.grad(learned_loss, theta_task, create_graph=True, retain_graph=True,
                                            allow_unused=True)
        fast_weights = sgd_step(theta_task, detector_grad, self.config.ADAPTIVE_LR)
        set_parameters(self.detector, fast_weights)
        post_adaptive_out = self.detector(NestedTensor(img[0:1], mask[0:1]))

        set_parameters(self.detector, theta)

        return {k: v.unsqueeze(0) for k, v in post_adaptive_out.items()}

    def forward(self, data, train=True):

        # reformat img and mask data
        b, s, c, w, h = data["frames"].shape
        img = data["frames"].view(b, s, c, w, h)
        mask = data["masks"].view(b, s, w, h)
        # reformat labels
        labels = []
        for i in range(b):
            labels.append([])
            for j in range(s):
                labels[i].append({
                    "labels": data["category_ids"][i][j],
                    "boxes": data["boxes"][i][j]
                })

        detector_losses = []
        supervisor_losses = []
        out_logits_list = []
        out_boxes_list = []

        theta = get_parameters(self.detector)

        for task in range(b):

            theta_task = clone_parameters(theta)

            # get supervisor grads
            detached_theta_task = detach_parameters(theta_task)
            set_parameters(self.detector, detached_theta_task)
            pre_adaptive_out = self.detector(NestedTensor(img[task], mask[task]))
            pre_adaptive_out["embedded_memory_features"] = pre_adaptive_out["embedded_memory_features"].unsqueeze(0)
            pre_adaptive_out["box_features"] = pre_adaptive_out["box_features"].unsqueeze(0)
            pre_adaptive_out["pred_logits"] = pre_adaptive_out["pred_logits"].unsqueeze(0)
            pre_adaptive_out["pred_boxes"] = pre_adaptive_out["pred_boxes"].unsqueeze(0)

            fusion_out = self.fusion(pre_adaptive_out)
            learned_loss = torch.norm(fusion_out["loss"])
            detector_grad = torch.autograd.grad(learned_loss, detached_theta_task, create_graph=True, retain_graph=True,
                                                allow_unused=True)
            fast_weights = sgd_step(detached_theta_task, detector_grad, self.config.ADAPTIVE_LR)
            set_parameters(self.detector, fast_weights)
            post_adaptive_out = self.detector(NestedTensor(img[task], mask[task]))

            # lowest loss policy experiment
            first_frame_out = {k: v[[0]] for k, v in post_adaptive_out.items()}
            gt_loss = self.criterion(first_frame_out, [labels[task][0]], background_c=0.1)
            gt_loss = gt_loss["loss_ce"] + 5 * gt_loss["loss_giou"] + 2 * gt_loss["loss_bbox"]
            iip = data["initial_image_path"][task]
            rew = torch.mean(gt_loss).item()
            if iip not in self.path_storage:
                self.path_storage[iip] = PathStorage()
            self.path_storage[iip].add_path(data["actions"][task][:4], rew)
            best_path = torch.tensor(self.path_storage[iip].get_label(data["actions"][task][:4]),
                                     dtype=torch.long, device=gt_loss.device)

            supervisor_loss = self.criterion(post_adaptive_out, labels[task], background_c=0.1)
            supervisor_loss["loss_path"] = F.cross_entropy(fusion_out["actions"].view(4, 4), best_path)
            supervisor_loss["policy_reward"] = gt_loss
            supervisor_losses.append({k: v.detach() for k, v in supervisor_loss.items()})
            supervisor_loss = supervisor_loss["loss_ce"] + 5 * supervisor_loss["loss_giou"] + \
                              2 * supervisor_loss["loss_bbox"] + supervisor_loss["loss_path"]
            supervisor_loss.backward()

            # get detector grads
            fast_weights = sgd_step(theta_task, detach_gradients(detector_grad), self.config.ADAPTIVE_LR)
            set_parameters(self.detector, fast_weights)

            ridx = random.randint(0, 4)
            post_adaptive_out = self.detector(NestedTensor(img[task][ridx:ridx+1], mask[task][ridx:ridx+1]))
            detector_loss = self.criterion(post_adaptive_out, labels[task][ridx:ridx+1], background_c=0.1)
            detector_losses.append({k: v.detach() for k, v in detector_loss.items()})
            detector_loss = detector_loss["loss_ce"] + 5 * detector_loss["loss_giou"] + 2 * detector_loss["loss_bbox"]
            detector_loss.backward()

            out_logits_list.append(post_adaptive_out["pred_logits"])
            out_boxes_list.append(post_adaptive_out["pred_boxes"])

        set_parameters(self.detector, theta)

        predictions = {"pred_logits": torch.stack(out_logits_list, dim=0), "pred_boxes": torch.stack(out_boxes_list, dim=0)}
        mean_detector_losses = {k.replace("loss", "loss_detector"):
                                    torch.mean(torch.stack([x[k] for x in detector_losses]))
                                for k, v in detector_losses[0].items()}
        mean_supervisor_losses = {k.replace("loss", "loss_supervisor"):
                                    torch.mean(torch.stack([x[k] for x in supervisor_losses]))
                                for k, v in supervisor_losses[0].items()}
        losses = mean_detector_losses
        losses.update(mean_supervisor_losses)

        return predictions, losses

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        # only train proposal generator of detector
        self.detector.train(mode)
        self.fusion.train(mode)
        return self

    def get_optimizer_groups(self, train_config):
        optim_groups = [
            {"params": list(self.decoder.parameters()), "weight_decay": 0.0},
            {"params": list(self.detector.parameters()), "weight_decay": 0.0},
        ]
        return optim_groups

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger

    def get_next_action(self, data):
        # reformat img and mask data
        b, s, c, w, h = data["frames"].shape
        img = data["frames"].view(b*s, c, w, h)
        mask = data["masks"].view(b*s, w, h)
        # reformat labels
        labels = []
        for i in range(b):
            labels.append([])
            for j in range(s):
                labels[i].append({
                    "labels": data["category_ids"][i][j],
                    "boxes": data["boxes"][i][j]
                })

        pre_adaptive_out = self.detector(NestedTensor(img, mask))
        pre_adaptive_out["embedded_memory_features"] = pre_adaptive_out["embedded_memory_features"].unsqueeze(0)
        pre_adaptive_out["box_features"] = pre_adaptive_out["box_features"].unsqueeze(0)
        pre_adaptive_out["pred_logits"] = pre_adaptive_out["pred_logits"].unsqueeze(0)
        pre_adaptive_out["pred_boxes"] = pre_adaptive_out["pred_boxes"].unsqueeze(0)

        fusion_out = self.fusion(pre_adaptive_out)

        return fusion_out['actions'][s-1].argmax(dim=-1).item()

