import torch
import torch.nn as nn
import math

from models.gpt import GPT
from models.detr_models.detr import MLP


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.img_feature_embedding = nn.Linear(config.IMG_FEATURE_SIZE, config.EMBEDDING_DIM)
        self.prediction_embedding = nn.Linear(config.BOX_EMB_SIZE + config.NUM_CLASSES + 5, config.EMBEDDING_DIM)
        self.model = GPT(config)
        self.box_decoder = MLP(config.OUTPUT_SIZE, 256, 4, 3)
        # self.logit_decoder = nn.Linear(config.OUTPUT_SIZE, config.NUM_CLASSES + 1)
        self.logit_decoder = self.box_decoder = MLP(config.OUTPUT_SIZE, 512, config.NUM_CLASSES + 1, 5)
        self.loss_decoder = MLP(config.OUTPUT_SIZE, 512, 1, 5)
        self.action_decoder = MLP(config.OUTPUT_SIZE, 512, 4, 5)
        self.action_tokens = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, 5, config.EMBEDDING_DIM),
                                                                   a=math.sqrt(5)))

    def forward(self, x):
        # fold data into sequence
        img_feature_embedding = self.img_feature_embedding(x["embedded_memory_features"].permute(0,1,3,4,2))
        preds = torch.cat((x["box_features"], x["pred_logits"], x["pred_boxes"]), dim=-1)
        prediction_embeddings = self.prediction_embedding(preds)
        b, s, p, n = prediction_embeddings.shape
        n_preds = prediction_embeddings.shape[1] * prediction_embeddings.shape[2]
        pad = torch.zeros((b, 2060, n), device=prediction_embeddings.device)
        seq = torch.cat((img_feature_embedding.reshape(b, -1, n),
                         prediction_embeddings.reshape(b, -1, n),
                         self.action_tokens.repeat(b,1,1).reshape(b, -1, n)), dim=1)
        pad[:, :seq.shape[1]] = seq
        y = self.model(pad)
        # unfold data
        y_preds = y[:, -(n_preds + 5):-5].reshape(b, s, p, -1)
        boxes = self.box_decoder(y_preds).sigmoid()
        logits = self.logit_decoder(y_preds)
        loss = self.loss_decoder(y_preds)
        actions = self.action_decoder(y[:, -5:-1].reshape(b, 4, -1))

        return {"seq": y_preds, "pred_boxes": boxes, "pred_logits": logits, "loss": loss, "actions": actions}
