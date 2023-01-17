from typing import Optional, Tuple

import numpy as np
from numpy import ndarray


class ConfusionMatrix:

    def __init__(self, num_classes: int = 80) -> None:
        self.num_classes = num_classes
        self.stats = []
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.size
        self.seen = 0

    def clean(self) -> None:
        self.stats.clear()
        self.seen = 0

    def add(self, pred: Optional[ndarray], label: Optional[ndarray]) -> None:
        # pred : x0, y0, x1, y1, score, label
        # label: x0, y0, x1, y1, label
        if pred is None:
            pred = np.empty((0, 6), dtype=np.float32)
        if label is None:
            label = np.empty((0, 5), dtype=np.float32)
        assert pred.shape[1] == 6 and label.shape[1] == 5
        self.seen += 1
        nl, npr = label.shape[0], pred.shape[0]
        correct = np.zeros((npr, self.niou), dtype=bool)
        if npr == 0:
            if nl:
                self.stats.append((correct, *np.zeros((2, 0)), label[:, -1]))
            return
        if nl:
            correct = self.process_batch(pred, label)
        self.stats.append((correct, pred[:, 4], pred[:, 5], label[:, -1]))

    def calculate(self) -> dict:
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = \
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = self.ap_per_class(*stats)
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=self.num_classes)
        per_class_info = dict(tp=tp,
                              fp=fp,
                              p=p,
                              precision=p,
                              recall=r,
                              f1=f1,
                              ap50=ap50)
        res = dict(detail=per_class_info,
                   num_bboxes=nt.sum(),
                   num_images=self.seen,
                   mean_precision=mp,
                   mean_recall=mr,
                   map50=map50,
                   map=map)
        return res

    def ap_per_class(
        self,
        tp: ndarray,
        conf: ndarray,
        pred_cls: ndarray,
        target_cls: ndarray,
        eps: float = 1e-16
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:

        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        pred_cls: ndarray

        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]

        px = np.linspace(0, 1, 1000)
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros(
            (nc, 1000))
        for ci, c in enumerate(unique_classes):
            i: ndarray = pred_cls == c
            n_l = nt[ci]
            n_p = i.sum()
            if n_p == 0 or n_l == 0:
                continue

            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            recall = tpc / (n_l + eps)

            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

            for j in range(tp.shape[1]):
                out = self.compute_ap(recall[:, j], precision[:, j])
                ap[ci, j], mpre, mrec = out
        f1 = 2 * p * r / (p + r + eps)

        i = self.smooth(f1.mean(0), 0.1).argmax()
        p, r, f1 = p[:, i], r[:, i], f1[:, i]
        tp = (r * nt).round()
        fp = (tp / (p + eps) - tp).round()
        return tp, fp, p, r, f1, ap, unique_classes.astype(int)

    def process_batch(self, detections: ndarray, labels: ndarray) -> ndarray:
        correct = np.zeros(
            (detections.shape[0], self.iouv.shape[0])).astype(bool)
        iou = self.box_iou(labels[:, :4], detections[:, :4])
        correct_class = labels[:, 4:5] == detections[:, 5]
        for i in range(self.niou):
            x = np.where((iou >= self.iouv[i]) & correct_class)
            if x[0].shape[0]:
                matches = np.concatenate(
                    (np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1],
                                                return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0],
                                                return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        res = np.array(correct, dtype=bool)
        return res

    @staticmethod
    def box_iou(box1: ndarray, box2: ndarray, eps: float = 1e-7) -> ndarray:
        a1, a2 = np.hsplit(box1, 2)
        a1, a2 = np.expand_dims(a1, 1), np.expand_dims(a2, 1)
        b1, b2 = np.hsplit(box2, 2)
        b1, b2 = np.expand_dims(b1, 0), np.expand_dims(b2, 0),

        inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)
        iou = inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
        return iou

    @staticmethod
    def compute_ap(recall: ndarray,
                   precision: ndarray,
                   method: str = 'interp') -> Tuple[ndarray, ndarray, ndarray]:
        mrec = np.concatenate((np.array([0.0], dtype=np.float32), recall,
                               np.array([1.0], dtype=np.float32)))
        mpre = np.concatenate((np.array([1.0], dtype=np.float32), precision,
                               np.array([0.0], dtype=np.float32)))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        if method == 'interp':
            x = np.linspace(0, 1, 101)
            ap = np.trapz(np.interp(x, mrec, mpre), x)
        else:
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap, mpre, mrec

    @staticmethod
    def smooth(y: ndarray, f: float = 0.05) -> ndarray:
        nf = round(len(y) * f * 2) // 2 + 1
        p = np.ones(nf // 2)
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
        return np.convolve(yp, np.ones(nf) / nf, mode='valid')
