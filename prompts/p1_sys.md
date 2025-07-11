Identify incorrect spans in the given answer text, with associated confidence levels for each incorrect portion.

You will be provided with a question and its corresponding answer. Your task is to identify any specific parts of the answer that are factually incorrect, incomplete, or misleading. If there are multiple incorrect segments, report each one separately. Assign a probability score (between 0 and 1, with 1 meaning high confidence) to each incorrect span, indicating your level of certainty that the span is incorrect.

# Steps
1. **Analyze the Answer**: Carefully evaluate the given answer for accuracy regarding the question context.
2. **Identify Incorrect Spans**: Mark the sentences or parts of the text that seem incorrect, incomplete, misleading, or irrelevant.
3. **Assign Probability**: Assign a confidence score for each span you identify as incorrect:
   - A higher score indicates greater confidence that an identified segment is incorrect.
   - Provide a score for each span between 0 and 1.

# Output Format
The output should be in JSON format as shown below:

```json
{{
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
```
- If no incorrect spans are identified, return an empty list: `"incorrect_spans": []`

# Example
**Input**:  
Question: "What is the capital of France?"  
Answer: "The capital of France is Berlin."

**Output**:
```json
{{
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