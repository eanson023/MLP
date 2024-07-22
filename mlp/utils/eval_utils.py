import numpy as np
import random


def compute_tiou(pred, gt, epsilon=1e-12):
    intersection = min(pred[1], gt[1]) - max(pred[0], gt[0])
    union = max(epsilon, max(pred[1], gt[1]) - min(pred[0], gt[0]))
    return max(0.0, float(intersection) / union)


def rank(pred, gt):
    for i, p in enumerate(pred):
        if gt[0] == p[0] and gt[1] == p[1]:
            return i + 1
    return 9999999


def get_evaluator(dt="didemo"):
    if dt in ["anet", "charades", "babel", "humanml3d", "charades_tslm"]:
        return TALLEvaluator()
    else:
        raise NotImplementedError("Not supported dataset type ({})".format(dt))


class TALLEvaluator(object):
    def __init__(self):
        self.tiou_threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
        # self.metrics = ["R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "mIoU"]
        self.metrics = ["R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "R1-0.9",
                        "R5-0.1", "R5-0.3", "R5-0.5", "R5-0.7", "R5-0.9", "mIoU", "R5-mIoU",
                        "R1-0.1_ckd", "R1-0.3_ckd", "R1-0.5_ckd", "R1-0.7_ckd", "R1-0.9_ckd",
                        "R5-0.1_ckd", "R5-0.3_ckd", "R5-0.5_ckd", "R5-0.7_ckd", "R5-0.9_ckd", "mIoU_ckd", "R5-mIoU_ckd",
                        ]
        self.duration = None

    def get_metrics(self):
        # "R5-0.1", "R5-0.3", "R5-0.5", "R5-0.7", "R5-0.9"
        return ["R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "R1-0.9"]

    def set_duration(self, duration=[]):
        if len(duration) == 0:
            self.duration = None
        else:
            self.duration = duration

    def eval_instance(self, pred, gt, topk):
        """ Compute Recall@topk at predefined tiou threshold for instance
        Args:
            pred: predictions of starting/end position; list of [start,end]
            gt: ground-truth of starting/end position; [start,end]
            topk: rank of predictions; int
        Return:
            correct: flag of correct at predefined tiou threshold [0.3,0.5,0.7]
        """
        correct = {str(tiou): 0 for tiou in self.tiou_threshold}
        find = {str(tiou): False for tiou in self.tiou_threshold}
        if len(pred) == 0:
            return correct

        if len(pred) > topk:
            pred = pred[:topk]

        best_tiou = 0
        for loc in pred:
            cur_tiou = compute_tiou(loc, gt)

            if cur_tiou > best_tiou:
                best_tiou = cur_tiou

            for tiou in self.tiou_threshold:
                if (not find[str(tiou)]) and (cur_tiou >= tiou):
                    correct[str(tiou)] = 1
                    find[str(tiou)] = True

        return correct, best_tiou

    def eval(self, preds, gts):
        """ Compute R@1 and R@5 at predefined tiou threshold [0.3,0.5,0.7]
        Args:
            pred: predictions consisting of starting/end position; list
            gt: ground-truth of starting/end position; [start,end]
        Return:
            correct: flag of correct at predefined tiou threshold [0.3,0.5,0.7]
        """
        num_instances = float(len(preds))
        all_rank1 = {"R1-" + str(tiou): 0 for tiou in self.tiou_threshold}
        all_rank5 = {"R5-" + str(tiou): 0 for tiou in self.tiou_threshold}
        r1miou, r5miou = 0, 0

        ii = 0
        pt_idx = random.randint(0, len(gts) - 1)
        for pred, gt_list in zip(preds, gts):
            correct_r1, iou_r1 = [], []
            correct_r5, iou_r5 = [], []
            for gt in gt_list:
                if ii == pt_idx:
                    if self.duration is not None:
                        print("pred: {}\tgt: {}\ttIoU: {:.4f}".format(
                            str(np.array(pred[0]) / self.duration[ii]),
                            str(np.array(gt) / self.duration[ii]),
                            compute_tiou(np.array(pred[0]).squeeze() / self.duration[ii],
                                        np.array(gt).squeeze() / self.duration[ii])
                        ))
                    else:
                        print("pred: {}\tgt: {}\ttIoU: {}".format(
                            str(pred[0]), str(gt), compute_tiou(np.array(pred[0]).squeeze(), gt)))

                # compute rank1
                correct, iou = self.eval_instance(pred, gt, topk=1)
                correct_r1.append(correct)
                iou_r1.append(iou)
                # compute rank5
                correct, iou = self.eval_instance(pred, gt, topk=5)
                correct_r5.append(correct)
                iou_r5.append(iou)
            
            iou = np.max(iou_r1)
            iou_idx = np.argmax(iou_r1)
            r1miou += iou
            for tiou in self.tiou_threshold:
                all_rank1["R1-" + str(tiou)] += correct_r1[iou_idx][str(tiou)]
            
            iou = np.max(iou_r5)
            iou_idx = np.argmax(iou_r5)
            r5miou += iou
            for tiou in self.tiou_threshold:
                all_rank5["R5-" + str(tiou)] += correct_r5[iou_idx][str(tiou)]

            ii += 1

        return all_rank1, all_rank5, r1miou, r5miou
