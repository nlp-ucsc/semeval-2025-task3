You are a reasoning assistant tasked with identifying factually incorrect spans in an answer based on the given context. Your task is to:
1. Carefully analyze the context and answer.
2. Identify specific spans in the answer that contradict the context.
3. Assign a confidence score to each identified span, based on the strength of the contradiction.
4. Provide reasoning steps to justify the identified incorrect spans.

Use the following examples as a guide:

---

# Example 1:

**Input**:

<context>
"Petra van Staveren won a gold medal in the 100 meter breaststroke at the 1984 Summer Olympics in Los Angeles."
</context>
<question>
"What medal did Petra van Staveren win, and when?"
</question>
<answer>
"Petra van Staveren won a silver medal in the 2008 Summer Olympics in Beijing, China."
</answer>

# Output: 
```json
{{
  "reasoning_steps": [
    "The context states Petra van Staveren won a gold medal in the 100 meter breaststroke at the 1984 Summer Olympics.",
    "The answer incorrectly states 'silver medal', which contradicts 'gold medal'.",
    "It also incorrectly states '2008 Summer Olympics', which contradicts '1984 Summer Olympics'.",
    "The location 'Beijing, China' also contradicts 'Los Angeles'."
  ],
  "incorrect_spans": [
    {{
      "text": "silver",
      "probability": 0.99
    }},
    {{
      "text": "2008",
      "probability": 0.99
    }},
    {{
      "text": "Beijing, China",
      "probability": 0.99
    }}
  ]
}}
```
---

# Example 2:
**Input**:

<context>
"The Elysiphale order is a group of fungi containing 10 genera."
</context>
<question>
"How many genera are in the Elysiphale order?"
</question>
<answer>
"The Elysiphale order contains 5 genera."
</answer>

# Output:
```json
{{
  "reasoning_steps": [
    "The context states that the Elysiphale order contains 10 genera.",
    "The answer incorrectly states '5 genera', which contradicts the '10 genera' in the context."
  ],
  "incorrect_spans": [
    {{
      "text": "5",
      "probability": 0.99
    }}
  ]
}}
```
---

# Example 3:
**Input**:

<context>
"All arachnids lack antennas as a distinguishing feature."
</context>
<question>
"Do arachnids have antennas?"
</question>
<answer>
"Yes, all arachnids have antennas. However, not all of them are visible to the naked eye."
</answer>

# Output:
```json
{{
  "reasoning_steps": [
    "The context states that all arachnids lack antennas as a distinguishing feature.",
    "The answer incorrectly states 'all arachnids have antennas', which contradicts the context."
  ],
  "incorrect_spans": [
    {{
      "text": "Yes",
      "probability": 0.6
    }},
    {{
      "text": "all arachnids",
      "probability": 0.8
    }},
    {{
      "text": "not all",
      "probability": 0.7
    }},
    {{
      "text": "naked eye",
      "probability": 0.9
    }}
  ]
}}
```
---

# Example 4:
**Input**:

<context>
"Chance the Rapper debuted in 2013 with the release of his mixtape 'Acid Rap.'"
</context>
<question>
"When did Chance the Rapper debut?"
</question>
<answer>
"Chance the rapper debuted in 2011."
</answer>

# Output:
```json
{{
  "reasoning_steps": [
    "The context states Chance the Rapper debuted in 2013 with the release of his mixtape 'Acid Rap'.",
    "The answer incorrectly states '2011', which contradicts '2013' in the context."
  ],
  "incorrect_spans": [
    {{
      "text": "2011",
      "probability": 0.91
    }}
  ]
}}
```
---

# Example 5:
**Input**:

<context>
"A sustainable city provides infrastructure to ensure equitable access to basic services, such as water, sanitation, and electricity."
</context>
<question>
"What defines a sustainable city?"
</question>
<answer>
"The UN's Sustainable City initiative defines a city as one that is: - Equipped with infrastructure and services to ensure sustainable and equitable access to a range of basic services, such as water, sanitation, and electricity; -."
</answer>

# Output:
```json
{{
  "reasoning_steps": [
    "The context defines a sustainable city as one that provides infrastructure to ensure equitable access to basic services like water, sanitation, and electricity.",
    "The answer introduces redundant and overly broad information ('services to ensure sustainable and equitable access...'), which does not directly contradict the context."
  ],
  "incorrect_spans": [
    {{
      "text": "infrastructure and services to ensure sustainable and equitable access to a range of basic services, such as water, sanitation, and electricity; -.",
      "probability": 0.89
    }}
  ]
}}
```
---
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