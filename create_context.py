import os
from collections import namedtuple
import click
from tqdm import tqdm
from langchain_community.utilities.you import YouSearchAPIWrapper
from langchain_community.retrievers import TavilySearchAPIRetriever

from concurrent.futures import ProcessPoolExecutor

from labeler.utils import load_jsonl
from tools.perplexity_tool import call_perplexity
from tools.translator import Translator
from tools.deepseek_tool import call_deepseek


QAPair = namedtuple("QAPair", ["id", "question", "answer"])

ALL_LANGS = ["ar", "de", "en", "es", "fi", "fr", "hi", "it", "sv", "zh"]


@click.group()
def main():
    pass


@main.command()
@click.option("--num-web-results", type=int, default=3)
@click.option("--lang", type=str, default="en")
@click.option("--split", type=str, default="val")
def ydc(num_web_results, lang, split):
    output_dir = f"data/context/{lang}-{split}.v2_ydc-3"
    os.makedirs(output_dir, exist_ok=True)

    ydc_search = YouSearchAPIWrapper(
        endpoint_type="search", num_web_results=num_web_results
    )
    qa_pairs = get_qa_pairs(lang, split)
    for qa_pair in tqdm(qa_pairs):
        results = ydc_search.raw_results(qa_pair.question)
        paragraphs = []
        for hit in results["hits"]:
            if hit["snippets"]:  # some hits don't have snippets
                paragraphs.append(" ".join(hit["snippets"]))

        with open(os.path.join(output_dir, f"{qa_pair.id}.context.txt"), "w") as f:
            f.write("\n\n".join(paragraphs))


@main.command()
@click.option("--num-web-results", type=int, default=5)
@click.option("--lang", type=str, default="en")
@click.option("--split", type=str, default="val")
def tavily(num_web_results, lang, split):
    output_dir = f"data/context/{lang}-{split}.v2_tavily-5"
    os.makedirs(output_dir, exist_ok=True)

    tavily_search = TavilySearchAPIRetriever(k=num_web_results)
    qa_pairs = get_qa_pairs(lang, split)
    for qa_pair in tqdm(qa_pairs):
        results = tavily_search.invoke(qa_pair.question)
        paragraphs = []
        for doc in results:
            if doc.page_content:  # some docs don't have page_content
                paragraphs.append(doc.page_content)

        with open(os.path.join(output_dir, f"{qa_pair.id}.context.txt"), "w") as f:
            f.write("\n\n".join(paragraphs))


def perplexity_single_process(lang, split, need_translation, target_language):
    version = "v2" if split == "val" else "v1"
    try:
        output_dir = f"data/context/{lang}-{split}.{version}_perplexity-sonar-pro"

        qa_pairs = get_qa_pairs(lang, split)
        for qa_pair in tqdm(qa_pairs, desc=f"Processing {lang}"):
            question = qa_pair.question
            if need_translation:
                output_dir = f"data/context/{lang}-{split}-translated-{target_language}.{version}_perplexity-sonar-pro"
                translator = Translator(model="gpt-4o", target_language=target_language)
                question = translator.translate(question)

            while True:
                try:
                    results = call_perplexity(question)
                    break
                except Exception as e:
                    print(f"Error processing language {lang}, error: {e}\nRetrying...")

            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{qa_pair.id}.context.txt"), "w") as f:
                f.write(results)
    except Exception as e:
        print(f"Error processing {lang}: {e}")


@main.command()
@click.option("--langs", type=str, default="en")
@click.option("--split", type=str, default="val")
@click.option("--need-translation", type=bool, default=False)
@click.option("--target-language", type=str, default=None)
def perplexity(langs, split, need_translation, target_language):
    if langs == "all":
        langs = ALL_LANGS
    else:
        langs = langs.split(",")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                perplexity_single_process,
                lang,
                split,
                need_translation,
                target_language,
            )
            for lang in langs
        ]
        for future in tqdm(futures):
            future.result()


def deepseek_r1_single_process(lang, split, need_translation, target_language):
    try:
        output_dir = f"data/context/{lang}-{split}.v2_deepseek-r1"

        qa_pairs = get_qa_pairs(lang, split)
        for qa_pair in tqdm(qa_pairs, desc=f"Processing {lang}"):
            question = qa_pair.question
            if need_translation:
                output_dir = f"data/context/{lang}-{split}-translated-{target_language}.v2_deepseek-r1"
                translator = Translator(model="gpt-4o", target_language=target_language)
                question = translator.translate(question)

            while True:
                try:
                    results = call_deepseek(question)
                    break
                except Exception as e:
                    print(f"Error processing language {lang}, error: {e}\nRetrying...")

            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{qa_pair.id}.context.txt"), "w") as f:
                f.write(results)
    except Exception as e:
        print(f"Error processing {lang}: {e}")


@main.command()
@click.option("--langs", type=str, default="en")
@click.option("--split", type=str, default="val")
@click.option("--need-translation", type=bool, default=False)
@click.option("--target-language", type=str, default=None)
def deepseek_r1(langs, split, need_translation, target_language):
    if need_translation:
        assert target_language is not None
    if langs == "all":
        langs = ALL_LANGS
    else:
        langs = langs.split(",")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                deepseek_r1_single_process,
                lang,
                split,
                need_translation,
                target_language,
            )
            for lang in langs
        ]
        for future in tqdm(futures):
            future.result()


@main.command()
@click.option("--lang", type=str, default="en")
@click.option("--split", type=str, default="val")
def qa_perplexity(lang, split):
    if split == "val":
        version = "v2"
    else:
        version = "v1"
    output_dir = f"data/context/{lang}-{split}.{version}_qa_perplexity-sonar-pro"
    os.makedirs(output_dir, exist_ok=True)

    qa_pairs = get_qa_pairs(lang, split)
    for qa_pair in tqdm(qa_pairs):
        query = f"{qa_pair.question}\nAnswer the above question in a way that can fact-check the following answer:\n{qa_pair.answer}"
        results = call_perplexity(query)

        with open(os.path.join(output_dir, f"{qa_pair.id}.context.txt"), "w") as f:
            f.write(results)


def get_qa_pairs(lang: str, split: str):
    version = "v2" if split == "val" else "v1"
    path = f"data/{split}/mushroom.{lang}-{split}.{version}.jsonl"
    data = load_jsonl(path)
    return [
        QAPair(example["id"], example["model_input"], example["model_output_text"])
        for example in data
    ]


if __name__ == "__main__":
    main()
