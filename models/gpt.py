"""
GPT model
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.EMBEDDING_DIM % config.NUM_HEADS == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        self.query = nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        self.value = nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        # regularization
        self.attn_drop = nn.Dropout(config.ATTENTION_PDROP)
        self.resid_drop = nn.Dropout(config.RESIDUAL_PDROP)
        # output projection
        self.proj = nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
        #                              .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))
        self.register_buffer("mask", torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)
                             .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))
        self.NUM_HEADS = config.NUM_HEADS

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.NUM_HEADS, C // self.NUM_HEADS).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.NUM_HEADS, C // self.NUM_HEADS).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.NUM_HEADS, C // self.NUM_HEADS).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.EMBEDDING_DIM)
        self.ln2 = nn.LayerNorm(config.EMBEDDING_DIM)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, 4 * config.EMBEDDING_DIM),
            nn.GELU(),
            nn.Linear(4 * config.EMBEDDING_DIM, config.EMBEDDING_DIM),
            nn.Dropout(config.RESIDUAL_PDROP),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # # input embedding stem
        # self.tok_emb = nn.Embedding(config.vocab_size, config.EMBEDDING_DIM)
        self.pos_emb = nn.Parameter(torch.zeros(1, 255, config.EMBEDDING_DIM))

        # self.seq_pos_embed = nn.Parameter(torch.zeros(1, 2060, config.EMBEDDING_DIM), requires_grad=False)
        self.seq_pos_embed = nn.Parameter(torch.zeros(1, 2060, config.EMBEDDING_DIM), requires_grad=True)
        self.embed_dim = config.EMBEDDING_DIM
        self.img_len = 19*19

        self.drop = nn.Dropout(config.EMBEDDING_PDROP)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.NUM_LAYERS)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.EMBEDDING_DIM)
        self.head = nn.Linear(config.EMBEDDING_DIM, config.OUTPUT_SIZE, bias=False)

        self.block_size = config.BLOCK_SIZE
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def init_pos_emb(self):
        img_sin_embed = get_2d_sincos_pos_embed(self.embed_dim // 2, int(self.img_len**.5))
        img_pos_embed = torch.zeros((1, self.img_len, self.embed_dim))
        img_pos_embed[:, :, :self.embed_dim // 2] = torch.from_numpy(img_sin_embed).float()

        seq_sin_embed = get_1d_sincos_pos_embed(self.embed_dim // 2, 11)
        seq_pos_embed = torch.zeros((1, 11, self.embed_dim))
        seq_pos_embed[:, :, self.embed_dim // 2:] = torch.from_numpy(seq_sin_embed).float()

        pred_sin_embed = get_1d_sincos_pos_embed(self.embed_dim // 2, 50)
        pred_pos_embed = torch.zeros((1, 50, self.embed_dim))
        pred_pos_embed[:, :, self.embed_dim // 2:] = torch.from_numpy(pred_sin_embed).float() + 0.2

        action_sin_embed = get_1d_sincos_pos_embed(self.embed_dim // 2, 5)
        action_pos_embed = torch.zeros((1, 5, self.embed_dim))
        action_pos_embed[:, :, :self.embed_dim // 2] = torch.from_numpy(action_sin_embed).float()

        pos_emb = torch.zeros((1, 2060, self.embed_dim))
        for i in range(5):
            pos_emb[:,(self.img_len+50)*i:(self.img_len+50)*i+self.img_len] = img_pos_embed + seq_pos_embed[:,i*2,:]
            pos_emb[:,(self.img_len+50)*i+self.img_len:(self.img_len+50)*(i+1)] = pred_pos_embed + seq_pos_embed[
                                                                                                   :,i*2+1,:]
        pos_emb[:, 2055:, :] = action_pos_embed[:, :, :] + seq_pos_embed[:, -1, :]

        self.seq_pos_embed.data.copy_(pos_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_optimizer_groups(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

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
        no_decay.add('pos_emb')

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

    def forward(self, seq):
        b, t = seq.shape[:2]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        position_embeddings = self.seq_pos_embed[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(seq + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


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
