import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_pr_curve(points, mAP, path):
    plt.plot([x["recall"] for x in points], [x["precision"] for x in points])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve for IOU=0.5, mAP={:.4f}".format(mAP))
    plt.savefig(path)


def draw_prediction_distribuion(tp, fp, path):
    plt.style.use('seaborn-deep')
    x = [p["confidence"] for p in tp]
    y = [p["confidence"] for p in fp]
    bins = np.linspace(0.0, 1.0, num=10)
    plt.hist([x, y], bins, label=['True Positives', 'False Positives'])
    plt.legend(loc='upper right')
    plt.title("Distribution of True Positive and False Positive Distribution Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Number of Predictions")
    plt.savefig(path)


def draw_preds_and_labels(images, preds, labels):
    img = images.get_images()[0, 0].detach().cpu().numpy()
    preds = preds.get_boxes()[0, 0]
    matched_labels = labels.get_matched_boxes()[0, 0]
    gt_labels = labels.get_boxes()[0, 0]
    for i in range(matched_labels.shape[0]):
        img = draw_box(img, matched_labels[i], (255, 0, 0), 2)
    for i in range(gt_labels.shape[0]):
        # img = draw_box(img, preds[i], (0, 0, 0), 1)
        img = draw_box(img, gt_labels[i], (0, 255, 0), 1)
    return torch.tensor(img)


def draw_box(img, box, color, thickness, label=False):
    b = box
    img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thickness)
    if label:
        img = cv2.putText(img, label, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                          color, 1)
    return img

