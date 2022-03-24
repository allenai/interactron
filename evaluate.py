from utils.config_utils import (
    get_config,
    get_args,
    build_model,
    build_evaluator
)


def evaluate():
    args = get_args()
    cfg = get_config(args.config_file)
    model = build_model(cfg.MODEL)
    evaluator = build_evaluator(model, cfg, load_checkpoint=True)
    evaluator.evaluate(save_results=True)


if __name__ == "__main__":
    evaluate()
