You are given a question and answer , context knowledge in text format.

Note: Keep the size of spans detected as small as possible.
Note: Give hallucination probabilitities for each span ranging from 0 to 1.
Note: Punctuations get the lowest probability.
Note: Give higher probability only if you are very confident about its incorrectness.

**Steps**
1. Identify the spans of text in the given answer that contradicts the with the facts in the given context knowledge.
2. Give output with probabilities of incorrectness for these spans.

Note: Keep the size of incorrect text spans detected as small as possible. It could even be a letter as part of a word, or a substring of a word.
Note: Give hallucination probabilitities for each span ranging from 0 to 1.
Note: Punctuations get the lowest probability.
Note: Give higher probability only if you are very confident about its incorrectness.