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


def evaluate_all():
    args = get_args()
    cfg = get_config(args.config_file)
    model = build_model(cfg.MODEL)
    evaluator = build_evaluator(model, cfg, load_checkpoint=False)

    results = {}
    checkpoints = glob.glob("training_results/interactron_random/06-26-2022:06:12:51/detectordetector*")
    checkpoints.sort()
    checkpoints = checkpoints[300:400]
    for checkpoint in tqdm.tqdm(checkpoints):
        weights = torch.load(checkpoint, map_location='cpu')['model']
        evaluator.model.load_state_dict(weights)
        mAP_50, mAP, tps, fps, fns = evaluator.evaluate(save_results=False)
        print(mAP_50, mAP)
        results[checkpoint] = {
            "mAP_50": mAP_50,
            "mAP": mAP,
            "tps": tps,
            "fps": fps,
            "fns": fns
        }

    with open('interactron_random_results_300_to_400.json', 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    evaluate_all()
