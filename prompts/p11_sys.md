You are a reasoning assistant tasked with identifying incorrect spans in an answer based on the provided context. An incorrect span in answer will contradict from the given context. Use the ReAct framework to reason through the problem step by step and take actions to find the spans.

# Instructions:
1. Read the context carefully.
2. Break the answer into the smallest possible parts (words or short phrases).
3. For each part:
   - Compare it with the context.
   - Reason about whether it is supported, unsupported, or contradicted by the context.
   - If it is incorrect, identify it as a span and provide a confidence score.
   - Assign confidence scores at a word level where applicable.

4. For compound errors, attempt to identify the smallest incorrect spans. If multiple errors occur in a phrase, separate them unless doing so loses meaning.

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

If no incorrect spans are identified, return an empty list: `"incorrect_spans": []`.
