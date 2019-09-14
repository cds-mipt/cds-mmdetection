import argparse
import os
import mmcv
import numpy as np
import pandas as pd
import re

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Params:

    def __init__(self, gt, iouType):
        """
        iouType - one of 'bbox', 'segm'
        """
        # список id изображений для подсчета метрик
        # пустой - использовать все
        self.imgIds = []

        self.classes = []

        # пороги IoU
        self.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        # площади объектов, для которых будут вычислeны метрики
        self.areas = {
            "all": [0 ** 2, 1e5 ** 2],
            "small": [0 ** 2, 32 ** 2],
            "medium": [32 ** 2, 96 ** 2],
            "large": [96 ** 2, 1e5 ** 2]
        }

        self.maxDets = [1000]

        # остальное, как правило, нет причин менять
        self.id_to_class = {cat_id: cat["name"] for cat_id, cat in gt.cats.items()}

        self.class_to_id = {cat["name"]: cat_id for cat_id, cat in gt.cats.items()}
        self.catIds = [self.class_to_id[cls] for cls in self.classes] or list(gt.cats.keys())
        self.useCats = 1
        self.iouType = iouType
        self.useSegm = None
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.areaRngLbl = list(self.areas.keys())
        self.areaRng = [self.areas[k] for k in self.areaRngLbl]
        if not self.imgIds:
            self.imgIds = sorted(gt.getImgIds())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser


def detection_metrics(coco_gt, coco_dt, params):
    def calk_cond_mean(s, area, cat_id=-1, iouThr="mean", maxDet=-1):
        p = coco_eval.params
        s = s[:, :, list(p.areaRngLbl).index(area), p.maxDets.index(maxDet)]
        if cat_id != -1:
            s = s[:, p.catIds.index(cat_id)]
        if iouThr != "mean":
            s = s[list(p.iouThrs).index(iouThr)]
        valid = s > -1
        return np.mean(s[valid]) if valid.any() else -1

    def AP(area, cat_id=-1, iouThr=None, maxDet=-1):
        s = coco_eval.eval['precision'].mean(axis=1)
        return calk_cond_mean(s, area, cat_id, iouThr, maxDet)

    def AR(area, cat_id=-1, iouThr=None, maxDet=-1):
        s = coco_eval.eval['recall']
        return calk_cond_mean(s, area, cat_id, iouThr, maxDet)

    def pr_curve(area, cat_id, iouThr, maxDet):
        p = coco_eval.params
        recall = p.recThrs
        ti = list(p.iouThrs).index(iouThr)
        ki = list(p.catIds).index(cat_id)
        ai = list(p.areaRngLbl).index(area)
        di = list(p.maxDets).index(maxDet)
        precision = coco_eval.eval['precision'][ti, :, ki, ai, di]
        return recall, precision

    coco_eval = COCOeval(coco_gt, coco_dt, params.iouType)
    coco_eval.params = params
    coco_eval.evaluate()
    coco_eval.accumulate()

    metrics = []
    p = coco_eval.params
    for cat_id in p.catIds:
        for area in p.areaRngLbl:
            for maxDet in p.maxDets:
                for iouThr in p.iouThrs:
                    ap = AP(area, cat_id, iouThr, maxDet)
                    ar = AR(area, cat_id, iouThr, maxDet)
                    recall, precision = pr_curve(area, cat_id, iouThr, maxDet)
                    metrics.append({
                        "class": p.id_to_class[cat_id],
                        "area": area,
                        "maxDet": maxDet,
                        "iouThr": iouThr,
                        "AP": ap,
                        "AR": ar,
                        "recall": list(recall),
                        "precision": list(precision)
                    })

    return pd.DataFrame(metrics)


def infer_model(config_file, checkpoint, tmp_file):
    os.system("python tools/test.py {} {} --json_out {}".format(config_file, checkpoint, tmp_file))


def evaluate(coco_gt, config, tmp_file_prefix):
    config_file = tmp_file_prefix + "_config.py"
    with open(config_file, "w") as f:
        f.write(config.text)

    metrics = []
    for epoch in range(1, 2):
        infer_model(config_file, os.path.join(config.work_dir, "latest.pth"), tmp_file_prefix)

        epoch_metrics = {"epoch": epoch}
        for iou_type in ["segm", "bbox"]:
            params = Params(coco_gt, iou_type)
            coco_dt = coco_gt.loadRes(tmp_file_prefix + ".{}.json".format(iou_type))

            df = detection_metrics(coco_gt, coco_dt, params)
            df = df.query("area == 'all'").set_index("iouThr")

            for metric_name in ["AP", "AR"]:
                for iou_thr in params.iouThrs:
                    k = iou_type + "_" + metric_name + "_" + str(iou_thr)
                    epoch_metrics[k] = df.loc[iou_thr, metric_name]
                k = iou_type + "_" + metric_name + "_0.5:0.95"
                epoch_metrics[k] = df[metric_name].mean()
        metrics.append(epoch_metrics)
        print(epoch_metrics)
        os.system("rm " + tmp_file_prefix + "*.json")
    return pd.DataFrame(metrics)


def update_test_conf(config, ann_file, img_prefix):
    config.data.test.ann_file = ann_file
    config.data.test.img_prefix = img_prefix

    lines = config.text.split("\n")
    where = None
    for i, line in enumerate(lines):
        line = "".join(line.strip().split())
        if line.startswith("data="):
            where = "data"
        if where == "data" and line.startswith("test="):
            where = "test"
        if where == "test" and line.startswith("ann_file="):
            where = "ann_file"
        if where == "ann_file" and line.startswith("img_prefix"):
            where = "img_prefix"

        if where == "ann_file":
            lines[i] = re.sub(r"\s*ann_file\s*=.*,", " " * 8 + "ann_file=\"" + ann_file + "\",", lines[i])
        if where == "img_prefix":
            lines[i] = re.sub(r"\s*img_prefix=.*,", " " * 8 + "img_prefix=\"" + img_prefix + "\",", lines[i])
    config.__dict__["_text"] = "\n".join(lines)
    print(config.text)


def main(args):
    tmp_file_prefix = "/tmp/" + args.config.split("/")[-1].split(".")[0]
    config = mmcv.Config.fromfile(args.config)

    coco_gt = COCO(config.data.test.ann_file)
    test_metrics = evaluate(coco_gt, config, tmp_file_prefix)
    test_metrics.to_csv(os.path.join(config.work_dir, "test_metrics.csv"), index=False)

    coco_gt = COCO(config.data.train.ann_file)
    update_test_conf(config, config.data.train.ann_file, config.data.train.img_prefix)
    train_metrics = evaluate(coco_gt, config, tmp_file_prefix)
    train_metrics.to_csv(os.path.join(config.work_dir, "train_metrics.csv"), index=False)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
