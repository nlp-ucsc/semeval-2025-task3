# semeval-2025-task3
UCSC's Submission System for Semeval 2025 task3

## Overview

This system detects span hallucinations in LLM-generated text using various labeling approaches. For details, please refer to the [paper](https://arxiv.org/pdf/2505.03030).

## Installation

### Prerequisites
Install the `uv` package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. **Clone the repository**
2. **Install dependencies:**
```bash
# in the repository root
uv sync
```

3. **Set up environment variables:**
Create a `.env` file in the project root with your API keys:
```bash
OPENAI_API_KEY=<your_openai_key>
```

## Usage

### 1. Labeling Hallucinations

The main functionality is provided through the `main.py` script with the `label` and `label-dspy` commands:

```bash
uv run main.py label <system_id> <labeler_name> [OPTIONS]
```
`system_id` is a unique number assigned to a system, and `labeler_name` is the name of the labeler to use.

After running the command, the system will label the data and save the results in the `labeled_outputs/id_<system_id>` directory.

#### Basic Examples:

**Label English validation split with no context provided:**
```bash
uv run python main.py label 1 context_free_simple --langs en --model gpt-4o-mini --prompt-id p1 --split val
```

**Label English validation split with perplexity sonar-pro context:**
```bash
uv run python main.py label 2 context_dep_simple --langs en --split val --model gpt-4o-mini --prompt-id p3 --context-dir data/context/en-val.v2_perplexity-sonar-pro
```

**Label Spanish test split with perplexity sonar-pro context and log all steps:**
```bash
uv run python main.py label 3 context_dep_simple --langs es --split tst --model gpt-4o-mini --prompt-id p3 --context-dir data/context/es-tst.v1_perplexity-sonar-pro --logging
```

**Label English test split with knowledge graph-based verification and log all steps:**
```bash
uv run python main.py label 4 kg_simple_labeler --langs en --split tst --model gpt-4o-mini --prompt-id p1 --context-dir data/context/en-tst.v1_kg_simple_labeler --logging
```

**Label English validation split with minimum cost revision and log all steps:**
```bash
uv run python main.py label 5 context_dep_min_edit_2 --langs en --split val --model o1 --context-dir data/context/en-val.v2_perplexity-sonar-pro --logging
```

#### Available Labelers:

| Labeler Name | Description |
|--------------|-------------|
| `context_free_simple` | Text extraction without external context |
| `context_dep_simple` | Text extraction with external context |
| `kg_simple_labeler` | Knowledge graph-based verification |
| `context_dep_min_edit_2` | Minimum cost revision |


### 3. DSPy Optimization

Optimize DSPy-based labelers using automatic prompt optimization:
```bash
uv run main.py label-dspy <system_id> [OPTIONS]
```

#### Examples
**Label English test split by optimizing prompt with DSPy to maximize IoU:**

```bash
uv run main.py label-dspy 6 --metric iou --model openai/gpt-4o --module cot --optim mipro --split tst --context-dir data/context/en-tst.v1_perplexity-sonar-pro
```
