You are given a question and answer , context knowledge in text format.

Note: Keep the size of spans detected as small as possible.
Note: Give hallucination probabilitities for each span ranging from 0 to 1.
Note: Punctuations get the lowest probability.
Note: Give higher probability only if you are very confident about its incorrectness.

**Steps**
1. Identify the substrings in the given answer that contradicts the facts in the given context knowledge.
2. Give output with probabilities of incorrectness for these spans.

Note: Keep the size of incorrect substring spans detected as small as possible. It could even be a single letter of a word.
Note: Give hallucination probabilitities for each span ranging from 0 to 1.
Note: If the span is not an entire word, assign a slightly less probability.
Note: Punctuations get the lowest probability.
Note: Give higher probability only if you are very confident about its incorrectness.