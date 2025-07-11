Identify all factually contradictory spans in the answer based on the context. Follow these rules:

# Steps to Follow:
1. Break down any multi-part contradictory phrase into its smallest components. For example:
   - If "silver medal in the 2008 Summer Olympics in Beijing, China" is incorrect, identify each part:
      - "silver"
      - "2008"
      - "Beijing, China"
2. Only label text as incorrect if it explicitly contradicts the context. Ignore irrelevant or non-contradictory information.
3. Assign a probability score to each incorrect span, ensuring that spans are granular and independent.

# Notes:
1. Factually Incorrect Spans: Only highlight substrings in the answer that explicitly contradict the context.
2. Ignore Unverifiable Claims: If a span cannot be verified from the context but does not contradict it, do not flag it.
3. Ensure that the spans are as short and specific as possible to capture the exact contradictory token.

# Output Format:
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
