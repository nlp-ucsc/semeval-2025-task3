import os
import random
import json
import re
import click
import glob
import sys
from scipy.stats import spearmanr
from collections.abc import Callable
from typing import Literal, Any
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

import dspy

from labeler.utils import load_jsonl, dump_jsonl
from labeler import get_labeler
import scorer
from labeler.utils import max_substring_match, train_test_split
from labeler.dspy_labeler import FindIncorrectSpans
from scorer import score_max_ious, load_jsonl_file_to_records
import numpy as np

load_dotenv()
ALL_VAL_LANGS = ["ar", "de", "en", "es", "fi", "fr", "hi", "it", "sv", "zh"]
ALL_TEST_LANGS = [
    "ar",
    "ca",
    "cs",
    "de",
    "en",
    "es",
    "eu",
    "fa",
    "fi",
    "fr",
    "hi",
    "it",
    "sv",
    "zh",
]
VAL_DIR = "data/val"
TEST_DIR = "data/tst"
OUTPUT_DIR = "labeled_outputs"


def parse_langs_and_paths(langs: str, split: str) -> tuple[list[str], list[str]]:
    version = "v2" if split == "val" else "v1"
    directory = VAL_DIR if split == "val" else TEST_DIR
    if langs == "all":
        langs_to_label = ALL_VAL_LANGS if split == "val" else ALL_TEST_LANGS
    else:
        langs_to_label = langs.split(",")
        assert all(
            lang in ALL_VAL_LANGS if split == "val" else ALL_TEST_LANGS
            for lang in langs_to_label
        ), "Invalid language"
    paths = [
        os.path.join(directory, f"mushroom.{lang}-{split}.{version}.jsonl")
        for lang in langs_to_label
    ]
    return langs_to_label, paths


def save_settings(output_dir: str, print_settings: bool = False, **kwargs):
    with open(os.path.join(output_dir, f"settings-{kwargs['langs']}.json"), "w") as f:
        json.dump(kwargs, f, indent=4)

    if print_settings:
        print(
            f"Labeling with settings:\n"
            f"  system_id: {kwargs['system_id']}\n"
            f"  Labeler: {kwargs['labeler_name']}\n"
            f"  Model: {kwargs['model']}\n"
            f"  prompt_id: {kwargs['prompt_id']}\n"
            f"  Other settings: {kwargs}"
        )


def score_labeled_outputs(output_dir: str):
    split = output_dir.split("/")[-1].split("-")[-1]
    score_file = os.path.join(output_dir, "scores.txt")
    f = open(score_file, "w")
    ious_all_langs = scorer.AverageVal()
    for pred_file in sorted(glob.glob(os.path.join(output_dir, "*.jsonl"))):
        lang = re.search(r"\.(\w{2})-", pred_file).group(1)
        if split == "val":
            ref_file = os.path.join(VAL_DIR, f"mushroom.{lang}-val.v2.jsonl")
        elif split == "tst":
            ref_file = os.path.join(TEST_DIR, f"mushroom.{lang}-tst.v1.jsonl")
        else:
            raise ValueError(f"Invalid split: {split}")
        
        ref = scorer.load_jsonl_file_to_records(ref_file)
        pred = scorer.load_jsonl_file_to_records(pred_file)
        iou, cor, _, _, _, _ = scorer.main(ref, pred, None)
        
        f.write(f"{lang} | IoU: {iou.mean():.4f} | Corr: {cor.mean():.4f}\n")
        ious_all_langs.update(iou.mean())
    
    f.write(f"{'-' * 28}\n")
    f.write(f"All Langs Mean IoU | {ious_all_langs.avg:.4f}\n")
    f.close()
    print(f"Saved scores to {score_file}")


@click.group()
def main():
    """CLI tool for labeling and scoring data."""
    pass


@main.command()
@click.argument("output_dir", type=str)
def score(output_dir: str):
    score_labeled_outputs(output_dir)


@main.command()
@click.argument("system_id", type=str)
@click.argument("labeler_name", type=str)
@click.option("--langs", type=str, default="en")
@click.option("--model", type=str, default="gpt-4o-mini")
@click.option("--prompt-id", type=str, default=None)
@click.option("--no-eval", is_flag=True, default=False)
@click.option("--split", type=str, default="val")
@click.option("--context-dir", type=str, default=None)
@click.option("--parse-span", type=str, default=None)
@click.option("--temperature", type=float, default=0.2)
@click.option("--threshold", type=float, default=0.5)
@click.option("--max_tokens", type=int, default=8192)
@click.option("--top_p", type=float, default=0.7)
@click.option("--seed", type=int, default=42)
@click.option("--logging", is_flag=True, default=False)
@click.option("--search", type=str, default=None)
def label(
    system_id: str,
    labeler_name: str,
    langs: str,
    model: str,
    prompt_id: str,
    no_eval: bool,
    split: str,
    threshold: float,
    context_dir: str = None,
    **kwargs,
):
    print(f"{kwargs = }")

    # determine the languages to label
    langs_to_label, paths = parse_langs_and_paths(langs, split)

    # verify all languages are present in the validation set
    for path in paths:
        assert os.path.exists(path), f"Path {path} does not exist"

    # create the output directory if it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR, f"id_{system_id}-{split}")
    os.makedirs(output_dir, exist_ok=True)

    # save the settings
    save_settings(
        output_dir,
        print_settings=True,
        system_id=system_id,
        labeler_name=labeler_name,
        model=model,
        prompt_id=prompt_id,
        context_dir=context_dir,
        langs=langs,
        threshold=threshold,
        command=f"uv run {' '.join(sys.argv)}",
        **kwargs,
    )
    print(f"{langs_to_label = }")

    # load the context
    context_map = {}
    if context_dir is not None:
        assert os.path.exists(
            context_dir
        ), f"Context directory {context_dir} does not exist"
        for context_file in glob.glob(os.path.join(context_dir, "*.txt")):
            id = os.path.basename(context_file).split(".", 1)[0]
            with open(context_file, "r", encoding="utf-8") as f:
                context_map[id] = f.read()

    # label the data
    labeler = get_labeler(labeler_name, model, prompt_id, **kwargs)

    for path, lang in zip(paths, langs_to_label):
        # configure logging
        if kwargs["logging"]:
            logging_file = os.path.join(output_dir, f"logging-{lang}.log")
            if os.path.exists(logging_file):
                os.remove(logging_file)
            logger.remove()
            logger.add(logging_file, level="INFO", format="{message}")

        data = load_jsonl(path)
        for example in tqdm(data, desc=f"Labeling {lang}"):
            soft_labels = labeler.label(
                example["model_input"],
                example["model_output_text"],
                context=context_map.get(example["id"], None),
                logger=logger if kwargs["logging"] else None,
            )
            hard_labels = [
                [sl["start"], sl["end"]]
                for sl in soft_labels
                if sl["prob"] >= threshold
            ]
            example["soft_labels"] = soft_labels
            example["hard_labels"] = hard_labels

        # save the labeled data
        output_path = os.path.join(output_dir, f"id_{system_id}.{lang}-{split}.jsonl")
        dump_jsonl(data, output_path)
        print(f"Saved labeled data to {output_path}")

    # score the data
    if not no_eval:
        score_labeled_outputs(output_dir)


class Program(dspy.Module):
    def __init__(self, module, model, temperature, max_tokens):
        self.lm = dspy.LM(
            model, temperature=temperature, max_tokens=max_tokens, cache=False
        )
        if module == "predict":
            self.find_incorrect_spans = dspy.Predict(FindIncorrectSpans)
        elif module == "cot":
            self.find_incorrect_spans = dspy.ChainOfThought(FindIncorrectSpans)
        else:
            raise ValueError(f"Invalid module: {module}")

    def __call__(self, **inputs) -> dspy.Prediction:
        pred: dspy.Prediction = self.find_incorrect_spans(**inputs)

        hard_labels = []
        soft_labels = []
        curr_idx = 0
        for span, confidence in zip(pred.incorrect_spans, pred.confidences):
            start = inputs["answer"].find(span, curr_idx)
            if start < 0:
                continue
            end = start + len(span)
            hard_labels.append((start, end))
            soft_labels.append({"start": start, "end": end, "prob": confidence})
            curr_idx = end
        pred.hard_labels = hard_labels
        pred.soft_labels = soft_labels

        return pred


@main.command()
@click.argument("system-id", type=str)
@click.option("--split", type=click.Choice(["val", "tst"]), default="val")
@click.option("--lang", type=str, default="en")
@click.option("--model", type=str, default="openai/gpt-4o-mini")
@click.option("--prompt-model", type=str, default="openai/gpt-4o")
@click.option(
    "--optim", type=click.Choice(["labeled_few_shot", "mipro"]), default="mipro"
)
@click.option(
    "--auto", type=click.Choice(["light", "medium", "heavy"]), default="light"
)
@click.option(
    "--metric",
    type=click.Choice(["iou", "corr", "max_iou", "iou_corr", "iou_corr_max_iou"]),
    default="iou",
)
@click.option("--module", type=click.Choice(["predict", "cot"]), default="predict")
@click.option(
    "--context-dir", type=str, default="data/context/en-val.v2_perplexity-sonar-pro"
)
@click.option("--temperature", type=float, default=0.2)
@click.option("--max-tokens", type=int, default=8192)
@click.option("--seed", type=int, default=42)
@click.option("--no-eval", is_flag=True, default=False)
def label_dspy(
    system_id: str,
    split: str,
    lang: str,
    model: str,
    metric: str,
    module: str,
    context_dir: str,
    no_eval: bool,
    **kwargs,
):
    assert (
        split in context_dir
    ), f"Context directory {context_dir} does not contain {split} data"

    # create the output directory if it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR, f"id_{system_id}-{split}")
    os.makedirs(output_dir, exist_ok=True)

    # save the settings
    save_settings(
        output_dir,
        dspy=True,
        system_id=system_id,
        split=split,
        langs=lang,
        model=model,
        metric=metric,
        module=module,
        context_dir=context_dir,
        command=f"uv run {' '.join(sys.argv)}",
        **kwargs,
    )

    val_context_dir = context_dir.replace("tst", "val").replace("v1", "v2")
    dataset = build_dataset("val", lang, val_context_dir)
    # random.shuffle(dataset)

    lm = dspy.LM(
        model,
        temperature=kwargs["temperature"],
        max_tokens=kwargs["max_tokens"],
        cache=False,
    )
    prompt_model = dspy.LM(kwargs["prompt_model"], temperature=1.0, cache=False)
    dspy.configure(lm=lm)

    match metric:
        case "iou":
            optimizer_metric = iou_metric
        case "corr":
            optimizer_metric = corr_metric
        case "max_iou":
            optimizer_metric = get_max_iou_metric()
        case "iou_corr":
            optimizer_metric = iou_corr_metric
        case "iou_corr_max_iou":
            optimizer_metric = get_iou_corr_max_iou_metric()

    match kwargs["optim"]:
        case "labeled_few_shot":
            optimizer = dspy.LabeledFewShot()
        case "mipro":
            optimizer = dspy.MIPROv2(
                metric=optimizer_metric,
                auto=kwargs["auto"],
                prompt_model=prompt_model,
            )
        case _:
            raise ValueError(f"Invalid optimizer: {kwargs['optim']}")

    labeled_examples = []
    if split == "val":
        for fold in range(2):
            if fold == 0:
                train_set, test_set = dataset[:25], dataset[25:]
            else:
                train_set, test_set = dataset[25:], dataset[:25]

            labeler = Program(
                module=module,
                model=model,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
            )

            if kwargs["optim"] == "mipro":
                labeler = optimizer.compile(
                    labeler,
                    trainset=train_set,
                    max_bootstrapped_demos=4,
                    max_labeled_demos=16,
                    requires_permission_to_run=False,
                    minibatch=False,
                )
            elif kwargs["optim"] == "labeled_few_shot":
                labeler = optimizer.compile(
                    labeler,
                    trainset=train_set,
                )
            model_save_path = os.path.join(output_dir, f"fold-{lang}_{fold}/")
            labeler.save(model_save_path, save_program=True)
            print(f"Saved model to {model_save_path}")

            for example in tqdm(test_set, desc=f"Labeling fold {fold}"):
                pred = labeler(**example.inputs())
                labeled_examples.append(
                    {
                        "id": example.id,
                        "lang": lang.upper(),
                        "model_input": example.question,
                        "model_output_text": example.answer,
                        "hard_labels": pred.hard_labels,
                        "soft_labels": pred.soft_labels,
                    }
                )
    else:
        train_set = dataset
        labeler = Program(
            module=module,
            model=model,
            temperature=kwargs["temperature"],
            max_tokens=kwargs["max_tokens"],
        )
        if kwargs["optim"] == "mipro":
            labeler = optimizer.compile(
                labeler,
                trainset=train_set,
                max_bootstrapped_demos=4,
                max_labeled_demos=16,
                requires_permission_to_run=False,
                minibatch=False,
            )
        elif kwargs["optim"] == "labeled_few_shot":
            labeler = optimizer.compile(
                labeler,
                trainset=train_set,
            )
        model_save_path = os.path.join(output_dir, f"model-{lang}/")
        labeler.save(model_save_path, save_program=True)
        print(f"Saved model to {model_save_path}")

        test_set = build_dataset(split, lang, context_dir)
        for example in tqdm(test_set, desc=f"Labeling"):
            pred = labeler(**example.inputs())
            labeled_examples.append(
                {
                    "id": example.id,
                    "lang": lang.upper(),
                    "model_input": example.question,
                    "model_output_text": example.answer,
                    "hard_labels": pred.hard_labels,
                    "soft_labels": pred.soft_labels,
                }
            )

    dump_jsonl(
        labeled_examples,
        os.path.join(output_dir, f"id_{system_id}.{lang}-{split}.jsonl"),
    )
    if split == "val" and not no_eval:
        score_labeled_outputs(output_dir)

    cost = sum(
        [x["cost"] for x in lm.history if x["cost"] is not None]
    )  # cost in USD, as calculated by LiteLLM for certain providers
    cost += sum([x["cost"] for x in prompt_model.history if x["cost"] is not None])
    print(f"Total cost: ${cost:.4f}")


@main.command()
@click.argument("system-id", type=str)
@click.option("--lang", type=str, default="en")
@click.option("--model-path", type=str)
@click.option("--context-dir", type=str, default=None)
def dspy_run_tst(system_id: str, lang, model_path: str, context_dir: str):
    output_dir = os.path.join(OUTPUT_DIR, f"id_{system_id}-tst")
    os.makedirs(output_dir, exist_ok=True)

    settings_path = os.path.join(os.path.dirname(model_path), f"settings-{lang}.json")
    settings = json.load(open(settings_path, "r", encoding="utf-8"))
    print(f"Using settings from {settings_path}")

    lm = dspy.LM(
        settings["model"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        cache=False,
    )
    dspy.configure(lm=lm)

    labeler = dspy.load(model_path)
    print(f"Loaded model from {model_path}")

    test_set = build_dataset("tst", settings["langs"], context_dir)
    labeled_examples = []
    for example in tqdm(test_set, desc=f"Labeling"):
        pred = labeler(**example.inputs())
        labeled_examples.append(
            {
                "id": example.id,
                "lang": settings["langs"].upper(),
                "model_input": example.question,
                "model_output_text": example.answer,
                "hard_labels": pred.hard_labels,
                "soft_labels": pred.soft_labels,
            }
        )

    dump_jsonl(
        labeled_examples,
        os.path.join(output_dir, f"id_{system_id}.{settings['langs']}-tst.jsonl"),
    )


def build_dataset(
    split: Literal["val", "tst"], lang: str, context_dir: str = None
) -> list[dspy.Example]:
    assert os.path.exists(
        context_dir
    ), f"Context directory {context_dir} does not exist"

    if split == "val":
        path = os.path.join(VAL_DIR, f"mushroom.{lang}-{split}.v2.jsonl")
    else:
        path = os.path.join(TEST_DIR, f"mushroom.{lang}-{split}.v1.jsonl")
    data = load_jsonl(path)

    context_map = {}
    for context_file in glob.glob(os.path.join(context_dir, "*.txt")):
        id = os.path.basename(context_file).split(".", 1)[0]
        with open(context_file, "r", encoding="utf-8") as f:
            context_map[id] = f.read()

    dataset = []
    if split == "val":
        for example in data:
            incorrect_spans = [
                example["model_output_text"][start:end]
                for start, end in example["hard_labels"]
            ]
            confidences = [
                round(compute_span_probability(start, end, example["soft_labels"]), 2)
                for start, end in example["hard_labels"]
            ]
            dataset.append(
                dspy.Example(
                    context=context_map[example["id"]],
                    question=example["model_input"],
                    answer=example["model_output_text"],
                    id=example["id"],
                    hard_labels=example["hard_labels"],
                    soft_labels=example["soft_labels"],
                    incorrect_spans=incorrect_spans,
                    confidences=confidences,
                ).with_inputs("context", "question", "answer")
            )
    else:
        for example in data:
            dataset.append(
                dspy.Example(
                    context=context_map[example["id"]],
                    question=example["model_input"],
                    answer=example["model_output_text"],
                    id=example["id"],
                ).with_inputs("context", "question", "answer")
            )

    return dataset


def compute_span_probability(hard_start, hard_end, soft_labels):
    """
    Computes a weighted average probability for the hard_label [hard_start, hard_end)
    by looking at all overlapping soft_labels.
    """
    total_prob = 0.0
    total_length = 0

    # for hard_idx in range(hard_start, hard_end):
    #     for sl in soft_labels:
    #         if hard_idx >= sl["start"] and hard_idx < sl["end"]:
    #             total_prob += sl["prob"]
    #             total_length += 1
    for sl in soft_labels:
        s_start, s_end, p = sl["start"], sl["end"], sl["prob"]
        overlap_start = max(hard_start, s_start)
        overlap_end = min(hard_end, s_end)
        overlap_len = overlap_end - overlap_start

        if overlap_len > 0:  # There's some overlap
            total_prob += p * overlap_len
            total_length += overlap_len

    if total_length == 0:
        return None  # No overlap found, no probability

    return total_prob / total_length


def iou_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    return iou(example.hard_labels, pred.hard_labels)


def iou(ref: list[tuple[int, int]], pred: list[tuple[int, int]]) -> float:
    ref_set = {idx for span in ref for idx in range(span[0], span[1])}
    pred_set = {idx for span in pred for idx in range(span[0], span[1])}
    if not ref_set and not pred_set:
        return 1.0
    return len(ref_set & pred_set) / len(ref_set | pred_set)


def corr_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    return corr(example.soft_labels, pred.soft_labels, len(example.answer))


def corr(ref: list[dict], pred: list[dict], length: int) -> float:
    ref_vec = [0.0] * length
    pred_vec = [0.0] * length
    for span in ref:
        for idx in range(span["start"], span["end"]):
            ref_vec[idx] = span["prob"]
    for span in pred:
        for idx in range(span["start"], span["end"]):
            pred_vec[idx] = span["prob"]
    if (
        len({round(flt, 8) for flt in pred_vec}) == 1
        or len({round(flt, 8) for flt in ref_vec}) == 1
    ):
        return float(
            len({round(flt, 8) for flt in ref_vec})
            == len({round(flt, 8) for flt in pred_vec})
        )
    return spearmanr(ref_vec, pred_vec).correlation


def get_max_iou_metric() -> Callable[[dspy.Example, dspy.Prediction, Any], float]:
    human_refs = load_jsonl_file_to_records("annotation/merged/en-val.jsonl")
    # breakpoint()
    human_refs_map = {example["id"]: example for example in human_refs}

    # breakpoint()
    def max_iou_metric(
        example: dspy.Example, pred: dspy.Prediction, trace=None
    ) -> float:
        # breakpoint()
        example_id = int(re.search(r"\d+", example.id).group())
        pred.id = example_id
        return score_max_ious(human_refs_map[example_id], pred)

    return max_iou_metric


def iou_corr_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    iou_score = iou(example.hard_labels, pred.hard_labels)
    corr_score = corr(example.soft_labels, pred.soft_labels, len(example.answer))
    return (iou_score + corr_score) / 2


def get_iou_corr_max_iou_metric() -> (
    Callable[[dspy.Example, dspy.Prediction, Any], float]
):
    max_iou_metric = get_max_iou_metric()

    def iou_corr_max_iou_metric(
        example: dspy.Example, pred: dspy.Prediction, trace=None
    ) -> float:
        iou_score = iou(example.hard_labels, pred.hard_labels)
        corr_score = corr(example.soft_labels, pred.soft_labels, len(example.answer))
        max_iou_score = max_iou_metric(example, pred)
        return (iou_score + corr_score + max_iou_score) / 3

    return iou_corr_max_iou_metric


if __name__ == "__main__":
    main()
