import torch
from utils.config_utils import (
    get_config,
    get_args,
    build_model,
    build_trainer,
    build_evaluator,
)


def train():
    args = get_args()
    cfg = get_config(args.config_file)
    model = build_model(cfg.MODEL)

    # checkpoint = torch.load('training_results/interactron_random/06-11-2022:22:09:54/detector.pt',
    #                         map_location='cpu')
    # model.load_state_dict(checkpoint['model'])

    evaluator = build_evaluator(model, cfg)
    trainer = build_trainer(model, cfg, evaluator=evaluator)
    trainer.train()


if __name__ == "__main__":
    train()
