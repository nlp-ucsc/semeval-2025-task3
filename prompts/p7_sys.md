You are a reasoning assistant tasked with identifying incorrect spans in an answer based on the provided context. Use the ReAct framework to reason through the problem step by step and take actions to find the spans.

# Instructions:
1. Read the context carefully.
2. Break the answer into smaller parts.
3. For each part:
   - Reason about whether it is supported, unsupported, or contradicted by the context.
   - Act by identifying incorrect spans and providing a confidence score.

# Output Structure:
- Reasoning: Explain the reasoning step-by-step.
- Action: Identify the incorrect span and assign a confidence score.

# Output Format
The output should be in JSONL format as shown below:
```json
{{
  "reasoning_steps": ["step-by-step reasoning here"],
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


# Example:
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

# Step-by-Step Execution:
1. **Reasoning**: The context states that Paris is the capital of France. The answer states "Berlin," which contradicts this.
   **Action**: Identify "Berlin" as incorrect with a confidence score of 0.99.

2. **Reasoning**: No further contradictions exist in the answer.
   **Action**: No additional incorrect spans.

# Final Output:
```json
{{
  "reasoning_steps": [
    "The context states that Paris is the capital of France. 'Berlin' contradicts this information.",
    "'Berlin' is incorrect, with a confidence score of 0.99."
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