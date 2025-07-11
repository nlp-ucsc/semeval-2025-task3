import pandas as pd
import re
from scipy.stats import spearmanr
import numpy as np

from collections import defaultdict
import argparse as ap


class AverageVal(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_human_soft_labels(human_labels):
    token_probabilities = {}
    annotator_count = len(human_labels)

    # Count how many annotators voted for each token
    for annotator, spans in human_labels.items():
        for span in spans:
            for token in range(span["start"], span["end"]):
                token_probabilities[token] = token_probabilities.get(token, 0) + 1

    # Normalize probabilities
    soft_labels = [
        {"start": token, "end": token + 1, "prob": count / annotator_count}
        for token, count in sorted(token_probabilities.items())
    ]
    return soft_labels


def recompute_hard_labels(soft_labels):
    """optionally, infer hard labels from the soft labels provided"""
    hard_labels = []
    prev_end = -1
    for start, end in (
        (lbl["start"], lbl["end"])
        for lbl in sorted(soft_labels, key=lambda span: (span["start"], span["end"]))
        if lbl["prob"] > 0.5
    ):
        if start == prev_end:
            hard_labels[-1][-1] = end
        else:
            hard_labels.append([start, end])
        prev_end = end
    return hard_labels


def load_jsonl_file_to_records(filename):
    """read data from a JSONL file and format that as a `pandas.DataFrame`.
    Performs minor format checks (ensures that soft_labels are present, optionally compute hard_labels on the fly).
    """
    df = pd.read_json(filename, lines=True)
    if "hard_labels" not in df.columns:
        df["hard_labels"] = df.soft_labels.apply(recompute_hard_labels)
    # adding an extra column for convenience
    if "human_labels" in df.columns:
        df["human_soft_labels"] = df.human_labels.apply(calculate_human_soft_labels)
        df["human_hard_labels"] = df.human_soft_labels.apply(recompute_hard_labels)
        # Transform {"start": x, "end": y} -> [x, y]
        df["human_labels"] = df["human_labels"].apply(
            lambda labels: {
                annotator: [[label["start"], label["end"]] for label in ranges]
                for annotator, ranges in labels.items()
            }
            if isinstance(labels, dict)
            else None
        )
    else:
        df["human_labels"] = None
        df["human_soft_labels"] = None
        df["human_hard_labels"] = None
    df["text_len"] = df.model_output_text.apply(len)
    df["id"] = df["id"].apply(lambda x: int(re.search(r"\d+", x).group()))
    df = df[
        [
            "id",
            "soft_labels",
            "hard_labels",
            "text_len",
            "human_labels",
            "human_hard_labels",
            "human_soft_labels",
        ]
    ]
    return df.sort_values("id").to_dict(orient="records")


def score_iou(ref_dict, pred_dict, human_ref_mode: bool = False):
    """computes intersection-over-union between reference and predicted hard labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    returns:
    the IoU, or 1.0 if neither the reference nor the prediction contain hallucinations
    """
    # ensure the prediction is correctly matched to its reference

    if human_ref_mode:
        hard_label_key = "human_hard_labels"
    else:
        hard_label_key = "hard_labels"

    assert ref_dict["id"] == pred_dict["id"]
    # convert annotations to sets of indices
    ref_indices = {idx for span in ref_dict[hard_label_key] for idx in range(*span)}
    pred_indices = {idx for span in pred_dict["hard_labels"] for idx in range(*span)}
    # avoid division by zero
    if not pred_indices and not ref_indices:
        return 1.0
    # otherwise compute & return IoU
    return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)


def score_max_ious(human_ref_dict, pred_dict):
    # breakpoint()
    assert human_ref_dict["id"] == pred_dict["id"]
    iou_records = []
    pred_indices = {idx for span in pred_dict["hard_labels"] for idx in range(*span)}
    for annotator, human_labels in human_ref_dict["human_labels"].items():
        human_indices = {idx for span in human_labels for idx in range(*span)}
        if not pred_indices and not human_indices:
            iou_records.append(1.0)
        else:
            iou_records.append(
                len(pred_indices & human_indices) / len(pred_indices | human_indices)
            )
    return max(iou_records) if iou_records else 0.0


def score_human_ious(human_ref_dict, pred_dict):
    assert human_ref_dict["id"] == pred_dict["id"]

    pred_indices = {idx for span in pred_dict["hard_labels"] for idx in range(*span)}

    iou_records = {}
    for annotator, human_labels in human_ref_dict["human_labels"].items():
        human_indices = {idx for span in human_labels for idx in range(*span)}

        if not pred_indices and not human_indices:
            iou = 1.0
        else:
            iou = len(pred_indices & human_indices) / len(pred_indices | human_indices)

        iou_records[annotator] = iou

    return iou_records


def group_ious_by_annotator(human_ref_dicts, pred_dicts):
    annotator_ious = defaultdict(list)
    for human_ref_dict, pred_dict in zip(human_ref_dicts, pred_dicts):
        ious = score_human_ious(human_ref_dict, pred_dict)
        for annotator, iou in ious.items():
            annotator_ious[annotator].append(iou)
    return annotator_ious


def score_cor(ref_dict, pred_dict, human_ref_mode: bool = False):
    """computes Spearman correlation between predicted and reference soft labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    returns:
    the Spearman correlation, or a binarized exact match (0.0 or 1.0) if the reference or prediction contains no variation
    """
    # ensure the prediction is correctly matched to its reference

    if human_ref_mode:
        soft_label_key = "human_soft_labels"
    else:
        soft_label_key = "soft_labels"

    assert ref_dict["id"] == pred_dict["id"]
    # convert annotations to vectors of observations
    ref_vec = [0.0] * ref_dict["text_len"]
    pred_vec = [0.0] * ref_dict["text_len"]
    for span in ref_dict[soft_label_key]:
        for idx in range(span["start"], span["end"]):
            ref_vec[idx] = span["prob"]
    for span in pred_dict["soft_labels"]:
        for idx in range(span["start"], span["end"]):
            pred_vec[idx] = span["prob"]
    # constant series (i.e., no hallucination) => cor is undef
    if (
        len({round(flt, 8) for flt in pred_vec}) == 1
        or len({round(flt, 8) for flt in ref_vec}) == 1
    ):
        return float(
            len({round(flt, 8) for flt in ref_vec})
            == len({round(flt, 8) for flt in pred_vec})
        )
    # otherwise compute Spearman's rho
    return spearmanr(ref_vec, pred_vec).correlation


def main(ref_dicts, pred_dicts, human_ref_dicts=None, output_file=None):
    assert len(ref_dicts) == len(pred_dicts)
    if human_ref_dicts is not None:
        assert len(pred_dicts) == len(human_ref_dicts)
    ious = np.array([score_iou(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    cors = np.array([score_cor(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    max_ious = (
        np.array([score_max_ious(r, d) for r, d in zip(human_ref_dicts, pred_dicts)])
        if human_ref_dicts
        else np.array([0])
    )
    human_ious = (
        group_ious_by_annotator(human_ref_dicts, pred_dicts)
        if human_ref_dicts
        else dict()
    )
    merged_human_iou = np.array(
        [
            score_iou(r, d, human_ref_mode=True)
            for r, d in zip(human_ref_dicts, pred_dicts)
        ]
        if human_ref_dicts
        else np.array([0])
    )
    merged_human_cor = np.array(
        [
            score_cor(r, d, human_ref_mode=True)
            for r, d in zip(human_ref_dicts, pred_dicts)
        ]
        if human_ref_dicts
        else np.array([0])
    )
    if output_file is not None:
        with open(output_file, "w") as ostr:
            print(f"IoU: {ious.mean():.8f}", file=ostr)
            print(f"Cor: {cors.mean():.8f}", file=ostr)
            print(f"Max IoU: {max_ious.mean():.8f}", file=ostr)
            for annotator, ious in human_ious.items():
                print(f"Annotation {annotator}: {np.mean(ious):.8f}", file=ostr)
            print(
                f"Human annotation mean IoU: {np.mean([np.mean(ious) for ious in human_ious.values()]):.8f}",
                file=ostr,
            )
            print(f"Merged human IoU: {merged_human_iou.mean():.8f}", file=ostr)
            print(f"Merged human Cor: {merged_human_cor.mean():.8f}", file=ostr)
    return ious, cors, max_ious, human_ious, merged_human_iou, merged_human_cor


if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument("ref_file", type=load_jsonl_file_to_records)
    p.add_argument("pred_file", type=load_jsonl_file_to_records)
    p.add_argument("human_ref_file", type=load_jsonl_file_to_records, default=None)
    p.add_argument("--output_file", default=None, type=str, required=False)
    a = p.parse_args()
    _ = main(a.ref_file, a.pred_file, a.human_ref_file, a.output_file)
