import yaml
import argparse
import os


ACTIONS = ["MoveAhead", "MoveBack", "RotateLeft", "RotateRight"]


class Config:

    def __init__(self, **entries):
        objectefied_entires = {}
        for entrie, value in entries.items():
            if type(value) is dict:
                objectefied_entires[entrie] = Config(**value)
            else:
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except:
                    pass
                objectefied_entires[entrie] = value
        self.__dict__.update(objectefied_entires)

    def dictionarize(self):
        fields = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                fields[k] = v.dictionarize()
            else:
                fields[k] = v
        return fields


def get_config(cfg):
    assert os.path.exists(cfg), "File {} does not exist".format(cfg)
    with open(cfg) as f:
        args = yaml.safe_load(f)
    return Config(**args)


def get_args():
    parser = argparse.ArgumentParser(description='Train Interactron Model')
    parser.add_argument('--config_file', type=str, required=True,
                        help='path to the configuration file for this training run')
    parser.add_argument('--devices', type=list, default='cpu', help='sum the integers (default: find the max)')

    args = parser.parse_args()
    return args


def build_model(args):
    arg_check(args.TYPE, ["detr", "detr_multiframe", "interactron_random", "interactron", "single_frame_baseline",
                          "five_frame_baseline", "adaptive"], "model")
    if args.TYPE == "single_frame_baseline":
        from models.single_frame_baseline import SingleFrameBaselineModel
        model = SingleFrameBaselineModel(args)
    elif args.TYPE == "five_frame_baseline":
        from models.five_frame_baseline import FiveFrameBaselineModel
        model = FiveFrameBaselineModel(args)
    elif args.TYPE == "adaptive":
        from models.adaptive import AdaptiveModel
        model = AdaptiveModel(args)
    elif args.TYPE == "detr":
        from models.detr import detr
        model = detr(args)
    elif args.TYPE == "detr_multiframe":
        from models.detr_multiframe import detr_multiframe
        model = detr_multiframe(args)
    elif args.TYPE == "interactron_random":
        from models.interactron_random import interactron_random
        model = interactron_random(args)
    elif args.TYPE == "interactron":
        from models.interactron import interactron
        model = interactron(args)
    return model


def build_trainer(model, args, evaluator=None):
    arg_check(args.TRAINER.TYPE, ["direct_supervision", "adaptive", "adaptive_interactive", "interactron_random",
                                  "interactron"], "supervisor")
    if args.TRAINER.TYPE == "direct_supervision":
        from engine.direct_supervision_trainer import DirectSupervisionTrainer
        trainer = DirectSupervisionTrainer(model, args, evaluator=evaluator)
    elif args.TRAINER.TYPE == "interactron_random":
        from engine.interactron_random_trainer import InteractronRandomTrainer
        trainer = InteractronRandomTrainer(model, args, evaluator=evaluator)
    elif args.TRAINER.TYPE == "interactron":
        from engine.interactron_trainer import InteractronTrainer
        trainer = InteractronTrainer(model, args, evaluator=evaluator)
    elif args.TRAINER.TYPE == "adaptive":
        from engine.adaptive_trainer import AdaptiveTrainer
        trainer = AdaptiveTrainer(model, args, evaluator=evaluator)
    elif args.TRAINER.TYPE == "adaptive_interactive":
        from engine.adaptive_interactive_trainer import AdaptiveInteractiveTrainer
        trainer = AdaptiveInteractiveTrainer(model, args, evaluator=evaluator)
    return trainer


def build_evaluator(model, args, load_checkpoint=False):
    arg_check(args.EVALUATOR.TYPE, ["random_policy_evaluator", "interactive_evaluator"], "evaluator")
    if args.EVALUATOR.TYPE == "random_policy_evaluator":
        from engine.random_policy_evaluator import RandomPolicyEvaluator
        evaluator = RandomPolicyEvaluator(model, args, load_checkpoint=load_checkpoint)
    elif args.EVALUATOR.TYPE == "interactive_evaluator":
        from engine.interactive_evaluator import InteractiveEvaluator
        evaluator = InteractiveEvaluator(model, args, load_checkpoint=load_checkpoint)
    return evaluator


def arg_check(arg, list, argname):
    assert arg in list, "{} is not a valid {}. Please select one from {}".format(arg, argname, list)


def iou(b1, b2):
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    i = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
    u = a1 + a2 - i
    return i / u


def compute_AP(precision, recall):
    p = precision
    r = recall
    return sum([r[0] * p[0]] + [(r[i]-r[i-1]) * ((p[i] + p[i-1])/2) for i in range(1, len(p))])
