import torch
import torch.nn as nn
import math
import numpy as np

from models.detr_models.detr import MLP
from models.detr_models.transformer import TransformerDecoderLayer, TransformerDecoder


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.img_feature_embedding = nn.Linear(config.IMG_FEATURE_SIZE, config.EMBEDDING_DIM)
        self.prediction_embedding = nn.Linear(config.BOX_EMB_SIZE + config.NUM_CLASSES + 5, config.EMBEDDING_DIM)
        self.box_decoder = MLP(config.OUTPUT_SIZE, 512, 4, 3)
        self.logit_decoder = nn.Linear(config.OUTPUT_SIZE, config.NUM_CLASSES + 1)
        self.loss_decoder = MLP(config.OUTPUT_SIZE, 512, 1, 3)
        self.action_decoder = MLP(config.OUTPUT_SIZE, 512, 4, 3)
        self.action_tokens = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, 5, config.EMBEDDING_DIM),
                                                                   a=math.sqrt(5)))
        # build transformer
        decoder_layer = TransformerDecoderLayer(config.EMBEDDING_DIM, config.NUM_HEADS, 2048, 0.1, "relu", False)
        decoder_norm = nn.LayerNorm(config.EMBEDDING_DIM)
        self.transformer = TransformerDecoder(decoder_layer, config.NUM_LAYERS, decoder_norm, return_intermediate=False)

        self.embed_dim = config.EMBEDDING_DIM
        self.img_len = 19 * 19

        self.pos_embed = nn.Parameter(torch.zeros(1, 1805, config.EMBEDDING_DIM), requires_grad=False)
        self.query_embed = nn.Parameter(torch.zeros(1, 255, config.EMBEDDING_DIM), requires_grad=True)
        self.init_pos_emb()

    def forward(self, x):
        # fold data into sequence
        img_feature_embedding = self.img_feature_embedding(x["embedded_memory_features"].permute(0, 1, 3, 4, 2))
        preds = torch.cat((x["box_features"], x["pred_logits"], x["pred_boxes"]), dim=-1)
        prediction_embeddings = self.prediction_embedding(preds)
        b, s, p, n = prediction_embeddings.shape
        # create padded sequences
        memory = torch.zeros((b, 5 * 19 * 19, n), device=prediction_embeddings.device)
        memory[:, :(s * 19 * 19)] = img_feature_embedding.reshape(b, -1, n)
        tgt = torch.zeros((b, 255, n))
        tgt[:, :(s * 50)] = prediction_embeddings.reshape(b, -1, n)
        tgt[:, 250:255] = self.action_tokens.repeat(b, 1, 1).reshape(b, -1, n)
        mask = torch.zeros((b, 5 * 19 * 19), dtype=torch.bool, device=x["box_features"].device)
        # pass sequence through model
        y = self.transformer(tgt.permute(1, 0, 2), memory.permute(1, 0, 2), memory_key_padding_mask=mask,
                             pos=self.pos_embed.permute(1, 0, 2), query_pos=self.query_embed.permute(1, 0, 2))
        # unfold data
        y_preds = y[:, :-5].reshape(b, s, p, -1)
        boxes = self.box_decoder(y_preds).sigmoid()
        logits = self.logit_decoder(y_preds)
        loss = self.loss_decoder(y_preds)
        actions = self.action_decoder(y[:, -5:-1].reshape(b, 4, -1))

        return {"seq": y_preds.squeeze(), "pred_boxes": boxes.squeeze(), "pred_logits": logits.squeeze(),
                "loss": loss, "actions": actions.squeeze()}

    def init_pos_emb(self):
        img_sin_embed = get_2d_sincos_pos_embed(self.embed_dim // 2, int(self.img_len**.5))
        img_pos_embed = torch.zeros((1, self.img_len, self.embed_dim))
        img_pos_embed[:, :, :self.embed_dim // 2] = torch.from_numpy(img_sin_embed).float()

        seq_sin_embed = get_1d_sincos_pos_embed(self.embed_dim // 2, 5)
        seq_pos_embed = torch.zeros((1, 5, self.embed_dim))
        seq_pos_embed[:, :, self.embed_dim // 2:] = torch.from_numpy(seq_sin_embed).float()

        pos_emb = torch.zeros((1, 1805, self.embed_dim))
        for i in range(5):
            pos_emb[:, self.img_len*i:self.img_len*(i+1)] = img_pos_embed + seq_pos_embed[:, i]

        self.pos_embed.data.copy_(pos_emb)


# Positional embeddings
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, n):
    grid = np.arange(n)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
