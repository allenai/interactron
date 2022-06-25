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
    evaluator = build_evaluator(model, cfg, load_checkpoint=True)

    results = {}
    for checkpoint in tqdm.tqdm(glob.glob("training_results/an/detectordetector*")):
        weights = torch.load(checkpoint, map_location='cpu')['model']
        model.load_state_dict(weights)
        mAP_50, mAP, tps, fps, fns = evaluator.evaluate(save_results=False)
        results[checkpoint] = {
            "mAP_50": mAP_50,
            "mAP": mAP,
            "tps": tps,
            "fps": fps,
            "fns": fns
        }

    with open('an_results.json', 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    evaluate_all()
