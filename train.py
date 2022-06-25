import torch
import random
import numpy
from utils.config_utils import (
    get_config,
    get_args,
    build_model,
    build_trainer,
    build_evaluator,
)


def train():
    random.seed(42)
    torch.manual_seed(42)
    numpy.random.seed(42)
    args = get_args()
    cfg = get_config(args.config_file)
    model = build_model(cfg.MODEL)
    evaluator = build_evaluator(model, cfg)
    trainer = build_trainer(model, cfg, evaluator=evaluator)
    trainer.train()


if __name__ == "__main__":
    train()
