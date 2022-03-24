# This code was copied from https://github.com/facebookresearch/detr/blob/main/models/detr.py
from .detr import build


def build_model(args):
    return build(args)
