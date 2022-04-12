import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.detr_models.detr import build
from models.detr_models.util.misc import NestedTensor
from models.transformer import Transformer
from models.learner import Learner
from utils.meta_utils import get_parameters, clone_parameters, sgd_step, set_parameters


class Decoder(nn.Module):

    def __init__(self, config, out_dim=4):
        super().__init__()
        parameter_list = [nn.Parameter(nn.init.kaiming_uniform_(torch.empty(512, config.OUTPUT_SIZE), a=math.sqrt(5)))]
        parameter_list += [nn.Parameter(nn.init.kaiming_uniform_(torch.empty(512, 512), a=math.sqrt(5))) for _ in range(3)]
        parameter_list.append(nn.Parameter(nn.init.kaiming_uniform_(torch.empty(out_dim, 512), a=math.sqrt(5))))
        self.weights = nn.ParameterList(parameter_list)

    def forward(self, x, grads=None, lr=1e-3):
        for i in range(len(self.weights) - 1):
            if grads is None:
                x = F.relu(F.linear(x, self.weights[i]))
            else:
                x = F.relu(F.linear(x, self.weights[i] - lr * grads[i]))
        if grads is None:
            x = F.linear(x, self.weights[-1])
        else:
            x = F.linear(x, self.weights[-1] - lr * grads[-1])
        return x

    def set_grad(self, grads):
        for i in range(len(self.weights)):
            grad = torch.mean(torch.stack([g[i] for g in grads]), dim=0)
            self.weights[i].grad = grad


def set_grad(model, grads):
    for i, p in enumerate(model.parameters()):
        if grads[0][i] is None:
            continue
        grad = torch.mean(torch.stack([g[i] for g in grads]), dim=0)
        p.grad = grad


class interactron_random(nn.Module):

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
        # self.decoder = Decoder(config, out_dim=config.NUM_CLASSES+1)
        self.decoder = Learner([
            ('linear', [512, config.OUTPUT_SIZE]),
            ('relu', [True]),
            # ('ln', [True]),
            ('bn', [5]),
            ('linear', [512, 512]),
            ('relu', [True]),
            # ('ln', [True]),
            ('bn', [5]),
            ('linear', [512, 512]),
            ('relu', [True]),
            # ('ln', [True]),
            ('bn', [5]),
            ('linear', [512, 512]),
            ('relu', [True]),
            # ('ln', [True]),
            ('bn', [5]),
            ('linear', [config.NUM_CLASSES+1, 512])
        ])
        self.logger = None
        self.mode = 'train'

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
        detector_grads = []
        supervisor_grads = []

        theta = get_parameters(self.detector)

        for task in range(b):
            # pre_adaptive_logits = self.decoder(detr_out["box_features"].clone().detach()[task:task+1])
            # pre_adaptive_logits = self.decoder(detr_out["box_features"].clone().detach()[task:task+1],
            #                                vars=None, bn_training=train)
            # in_seq = {
            #     "pred_logits": pre_adaptive_logits,
            #     "pred_boxes": detr_out["pred_boxes"][task:task+1].clone().detach(),
            #     "embedded_memory_features": detr_out["embedded_memory_features"][task:task+1].clone().detach(),
            #     "box_features": detr_out["box_features"][task:task+1].clone().detach(),
            # }

            theta_task = clone_parameters(theta)
            set_parameters(self.detector, theta_task)

            pre_adaptive_out = self.detector(NestedTensor(img[task][1:], mask[task][1:]))
            gt_losses = self.criterion(pre_adaptive_out, labels[task][1:], background_c=0.1)
            gt_loss = gt_losses["loss_ce"] + 5 * gt_losses["loss_bbox"] + 2 * gt_losses["loss_giou"]
            grad = torch.autograd.grad(gt_loss, theta_task)

            fast_weights = sgd_step(theta_task, grad, 0.001)
            set_parameters(self.detector, fast_weights)

            # learned_loss = torch.norm(self.fusion(in_seq)["loss"])
            # task_detr_full_out = {}
            # for key in detr_out:
            #     task_detr_full_out[key] = detr_out[key][task].reshape(1 * s, *detr_out[key].shape[2:])[1:]
            # full_in_seq = {}
            # for key in in_seq:
            #     full_in_seq[key] = in_seq[key].view(1 * s, *in_seq[key].shape[2:])[1:]
            #
            # gt_loss = self.criterion(full_in_seq, labels[task][1:], background_c=0.1)
            # grad = torch.autograd.grad(gt_loss["loss_ce"], self.decoder.parameters())
            # fast_weights = list(map(lambda p: p[1] - 1e-2 * p[0], zip(grad, self.decoder.parameters())))
            #
            # post_adaptive_logits = self.decoder(detr_out["box_features"].clone().detach()[task:task+1],
            #                                 fast_weights, bn_training=train)
            #
            # for k in range(5):
            #     in_seq = {
            #         "pred_logits": post_adaptive_logits,
            #         "pred_boxes": detr_out["pred_boxes"][task:task + 1].clone().detach(),
            #         "embedded_memory_features": detr_out["embedded_memory_features"][task:task + 1].clone().detach(),
            #         "box_features": detr_out["box_features"][task:task + 1].clone().detach(),
            #     }
            #     full_in_seq = {}
            #     for key in in_seq:
            #         full_in_seq[key] = in_seq[key].view(1 * s, *in_seq[key].shape[2:])[1:]
            #
            #     fast_loss = self.criterion(full_in_seq, labels[task][1:], background_c=0.1)
            #     grad = torch.autograd.grad(fast_loss["loss_ce"], fast_weights)
            #     fast_weights = list(map(lambda p: p[1] - 1e-2 * p[0], zip(grad, fast_weights)))
            #     post_adaptive_logits = self.decoder(detr_out["box_features"].clone().detach()[task:task + 1],
            #                                         fast_weights, bn_training=train)
            #
            # out_seq = {
            #     "pred_logits": post_adaptive_logits,
            #     "pred_boxes": detr_out["pred_boxes"][task:task+1].clone().detach()
            # }

            # out_seq = {
            #     "pred_logits": pre_adaptive_logits,
            #     "pred_boxes": detr_out["pred_boxes"][task:task+1].clone().detach()
            # }

            # full_out_seq = {}
            # for key in out_seq:
            #     full_out_seq[key] = out_seq[key].view(1 * s, *out_seq[key].shape[2:])[1:]
            # for key in out_seq:
            #     out_seq[key] = out_seq[key].view(1 * s, *out_seq[key].shape[2:])[:]
            task_detr_out = {}
            # for key in detr_out:
            #     task_detr_out[key] = detr_out[key][task].reshape(1 * s, *detr_out[key].shape[2:])[:]
            # task_detr_full_out = {}
            # for key in detr_out:
            #     task_detr_full_out[key] = detr_out[key][task].reshape(1 * s, *detr_out[key].shape[2:])[1:]

            post_adaptive_out = self.detector(NestedTensor(img[task][0:1], mask[task][0:1]))

            detector_loss = self.criterion(post_adaptive_out, labels[task][0:1], background_c=0.1)
            detector_losses.append(detector_loss)
            # supervisor_loss = self.criterion(full_out_seq, labels[task][1:], background_c=0.1)
            # supervisor_losses.append(supervisor_loss)
            out_logits_list.append(post_adaptive_out["pred_logits"])
            out_boxes_list.append(post_adaptive_out["pred_boxes"])

            # print(gt_loss["loss_ce"].item(), gt_loss["cardinality_error"].item(),
            #       detector_loss["loss_ce"].item(), detector_loss["cardinality_error"].item())

            # supervisor_grad = torch.autograd.grad(
            #     supervisor_loss["loss_ce"],
            #     self.fusion.parameters(),
            #     retain_graph=True,
            #     allow_unused=True
            # )
            # detector_grad = torch.autograd.grad(
            #     detector_loss["loss_ce"],
            #     self.decoder.parameters(),
            #     retain_graph=True,
            #     allow_unused=True,
            # )
            # supervisor_grads.append(supervisor_grad)
            # detector_grads.append([dg.detach() for dg in detector_grad])

        # set_grad(self.decoder, detector_grads)
        # set_grad(self.fusion, supervisor_grads)

        set_parameters(self.detector, theta)

        predictions = {"pred_logits": torch.stack(out_logits_list, dim=0), "pred_boxes": torch.stack(out_boxes_list, dim=0)}
        mean_detector_losses = {k.replace("loss", "loss_detector"):
                                    torch.mean(torch.stack([x[k] for x in detector_losses]))
                                for k, v in detector_losses[0].items()}
        # mean_supervisor_losses = {k.replace("loss", "loss_supervisor"):
        #                             torch.mean(torch.stack([x[k] for x in supervisor_losses]))
        #                         for k, v in supervisor_losses[0].items()}
        losses = mean_detector_losses
        # losses.update(mean_supervisor_losses)
        return predictions, losses

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        # only train proposal generator of detector
        self.detector.train(mode)
        self.fusion.train(mode)
        self.decoder.train(mode)
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

