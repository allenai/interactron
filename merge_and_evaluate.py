import tqdm
import json
import glob
import torch
from utils.config_utils import (
    get_config,
    get_args,
    build_model,
    build_evaluator
)


# various joining strageties
def mean_join(weights):
    window = 50
    joined_weight = torch.mean(torch.stack(weights[-window:], dim=0), dim=0)
    return joined_weight


def linear_weighted_join(weights):
    window = 50
    scales = torch.arange(1, window+1, 1) * (1/window)
    scales = scales.view(-1, *(1 for _ in range(len(weights[0].shape))))
    joined_weight = torch.sum(torch.stack(weights[-window:], dim=0) * scales, dim=0) / torch.sum(scales)
    return joined_weight


def exponential_weighted_join(weights):
    window = 50
    scales = torch.pow(torch.ones(window) * 0.95, torch.arange(window, 0, -1))
    scales = scales.view(-1, *(1 for _ in range(len(weights[0].shape))))
    joined_weight = torch.sum(torch.stack(weights[-window:], dim=0) * scales, dim=0) / torch.sum(scales)
    return joined_weight


def evaluate_all():
    args = get_args()
    cfg = get_config(args.config_file)
    model = build_model(cfg.MODEL)
    evaluator = build_evaluator(model, cfg, load_checkpoint=False)

    with open('selections-multi-frame.json') as fp:
        model_groups = json.load(fp)

    results = {}
    for group_name, group_weight_paths in tqdm.tqdm(model_groups.items()):
        # join the weights
        models = [torch.load(mp, map_location='cpu')['model'] for mp in group_weight_paths]
        joined_weights = {}
        for k in models[0].keys():
            joined_weights[k] = mean_join([m[k] for m in models])

        evaluator.model.load_state_dict(joined_weights)
        mAP_50, mAP, tps, fps, fns = evaluator.evaluate(save_results=False)
        print(group_name, "mAP_50:", mAP_50, "mAP:", mAP)
        results[group_name] = {
            "mAP_50": mAP_50,
            "mAP": mAP,
            "tps": tps,
            "fps": fps,
            "fns": fns
        }

    with open('multi-frame_results.json', 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    evaluate_all()
