import torch
import torch.nn as nn
import math

from models.gpt import GPT
from models.detr_models.detr import MLP


class MLP2(nn.Module):

    def __init__(self, in_dim, emb_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, x):
        return self.model(x)


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.img_feature_embedding = nn.Linear(config.IMG_FEATURE_SIZE, config.EMBEDDING_DIM)
        self.prediction_embedding = nn.Linear(config.BOX_EMB_SIZE + config.NUM_CLASSES + 5, config.EMBEDDING_DIM)
        self.model = GPT(config)
        self.box_decoder = MLP(config.OUTPUT_SIZE, 256, 4, 3)
        self.logit_decoder = nn.Linear(config.OUTPUT_SIZE, config.NUM_CLASSES + 1)
        self.loss_decoder = MLP(config.OUTPUT_SIZE, 512, 1, 3)
        self.action_decoder = MLP(config.OUTPUT_SIZE, 512, 4, 3)
        self.action_tokens = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, 5, config.EMBEDDING_DIM),
                                                                   a=math.sqrt(5)))

    def forward(self, x):
        # fold data into sequence
        img_feature_embedding = self.img_feature_embedding(x["embedded_memory_features"].permute(0, 1, 3, 4, 2))
        preds = torch.cat((x["box_features"], x["pred_logits"], x["pred_boxes"]), dim=-1)
        prediction_embeddings = self.prediction_embedding(preds)
        b, s, p, n = prediction_embeddings.shape
        n_preds = prediction_embeddings.shape[1] * prediction_embeddings.shape[2]
        seq = torch.cat((img_feature_embedding.reshape(b, -1, n),
                         prediction_embeddings.reshape(b, -1, n),
                         self.action_tokens.repeat(b, 1, 1).reshape(b, -1, n)), dim=1)
        y = self.model(seq)
        # unfold data
        y_preds = y[:, -(n_preds + 5):-5].reshape(b, s, p, -1)
        boxes = self.box_decoder(y_preds).sigmoid()
        logits = self.logit_decoder(y_preds)
        loss = self.loss_decoder(y_preds)
        actions = self.action_decoder(y[:, -5:-1].reshape(b, 4, -1))

        return {"seq": y_preds.squeeze(), "pred_boxes": boxes.squeeze(), "pred_logits": logits.squeeze(),
                "loss": loss, "actions": actions.squeeze()}

    def get_optimizer_groups(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('model.pos_emb')
        no_decay.add('action_tokens')
        no_decay.add('model.seq_pos_embed')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.WEIGHT_DECAY},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups
