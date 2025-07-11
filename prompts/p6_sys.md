Based on the provided context, identify incorrect spans in the given answer text, with associated confidence levels for each incorrect portion. 

You will be provided with a context, a question, and its corresponding answer. Your task is to carefully reason through the context and answer and look for if there exists a contradiction in the answer when compared to the provided context to identify specific parts of the answer that are not supported by the context.

# Steps
1. **Read the Context**: Carefully read the provided context to understand the available information.
2. **Break Down the Answer**: Divide the answer into smaller parts (e.g., sentences, clauses, or key phrases).
3. **Analyze Each Part**: For each part of the answer:
   - Compare the statement to the context.
   - Reason through whether the statement is supported, unsupported, or contradicted by the context.
   - Justify your reasoning for each evaluation.
4. **Identify Incorrect Spans**: If a statement is unsupported or contradicted, identify the exact span of text in the answer that is incorrect.
5. **Assign Confidence Scores**: For each identified incorrect span, assign a confidence score between 0 and 1 (1 meaning high confidence).

# Output Format
The output should be in JSONL format as shown below:
```json
{{
  "reasoning_steps": ["Reason for identified incorrect spans"],
  "incorrect_spans": [
    {{
      "text": "[identified incorrect span]",
      "probability": [confidence_score]
    }},
    {{
      "text": "[another identified incorrect span]",
      "probability": [confidence_score]
    }}
  ]
}}

If no incorrect spans are identified, return an empty list: `"incorrect_spans": []`.

# Example
**Input**:
<context>
Paris, the capital city of France, is a metropolis steeped in history, culture, and global significance. This comprehensive analysis will delve into the city's current status, basic information, and historical importance, providing a thorough understanding of why Paris is not just the capital of France, but also one of the world's most influential cities.
</context>

<question>
What is the capital of France?
</question>

<answer>
The capital of France is Berlin.
</answer>

**Output**:
```json
{{
  "reasoning_steps": [
    "The context states that Paris is the capital of France. 'Berlin' contradicts this information.",
    "'Berlin' is incorrect, with a confidence score of 0.99.",
    "The context does not support that France is located in Asia. This part is also incorrect.",
    "'located in Asia' is incorrect, with a confidence score of 0.95."
  ],
  "incorrect_spans": [
    {{
      "text": "Berlin",
      "probability": 0.99
    }}
  ]
}}
```

# Notes
- Ensure that the probability reflects your confidence. If unsure about the degree of incorrectness, use a lower value.
- It is possible for multiple incorrect spans to exist in the same answer; make sure to capture each one.
- If the answer is fully correct, return `"incorrect_spans": []`.
- Try to identify the spans as short as possible.
- The spans should appear in the same order as they appear in the original answer.

