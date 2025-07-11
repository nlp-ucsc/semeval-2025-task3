import json
import random

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def dump_jsonl(data: list[dict], path: str):
    with open(path, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

def transform_seq(a, b):
    """
    Transform list 'a' into list 'b' with minimal edit operations
    (delete, insert, substitute). Return both the minimum number of 
    operations and a list describing each operation.

    Parameters:
    -----------
    a : list
        The source list.
    b : list
        The target list.

    Returns:
    --------
    min_operations : int
        The minimum number of operations required to transform 'a' into 'b'.
    steps : list of str
        A list of strings, each describing one edit operation in sequence.
    """
    n = len(a)
    m = len(b)

    # dp[i][j] will hold the minimum edit distance (cost) to convert
    # a[:i] (length i) into b[:j] (length j).
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Initialize the first row and column:
    # - dp[i][0] means transforming a[:i] into an empty list => i deletions
    # - dp[0][j] means transforming an empty list into b[:j] => j insertions
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    # Fill in the dp table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                # If last elements are the same, no additional cost is needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Otherwise, consider the cost of insert, delete, or substitute
                dp[i][j] = 1 + min(dp[i - 1][j],     # Delete from 'a'
                                   dp[i][j - 1],     # Insert into 'a'
                                   dp[i - 1][j - 1]) # Substitute in 'a'

    # The minimum number of operations is at dp[n][m]
    min_operations = dp[n][m]

    # Backtrack to find the actual operations
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        # If we're in a situation where characters match or no change needed
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            # Move diagonally, no operation needed
            i -= 1
            j -= 1
        # If substitution was the chosen operation
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            steps.append(("sub", a[i-1]))
            i -= 1
            j -= 1
        # If insertion was the chosen operation
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            steps.append(("ins", b[j-1]))
            j -= 1
        # If deletion was the chosen operation
        else:
            steps.append(("del", a[i-1]))
            i -= 1

    # The steps we collected go from end to start, reverse them
    steps.reverse()

    return min_operations, steps


def find_change_spans(orig: str, target: str) -> list[dict[str, int | float]]:
    _, edit_steps = transform_seq(orig.split(), target.split())
    result = []
    for op, span in edit_steps:
        if op == "sub" or op == "del":
            start = orig.find(span)
            end = start + len(span)
            result.append({"start": start, "end": end, "prob": 1.0})
    for front, back in zip(result[:-1], result[1:]):
        if front["end"] == back["start"] - 1 and orig[front["end"]].isspace():
            front["end"] = back["start"]
    return result, edit_steps


def max_substring_match(a: str, b: str) -> tuple[int, int]:
    """
    Finds the start and end (non-inclusive) indices in 'a' of the
    *longest contiguous matching region* when string 'b' is aligned
    at every possible offset in 'a'. 
    
    Unlike the previous version, this function returns the actual 
    boundaries of the *consecutive matches*, so the returned substring 
    in 'a' may be shorter than 'b' if there are mismatches.
    
    Example:
        a = 'abxabcabcabcxyz'
        b = 'abcabz'
        
        The best contiguous match is 5 characters long, at a[3:8] = 'abcab'
        aligned against the first 5 chars of 'b' = 'abcab'.
        
        So the function returns (3, 8), i.e. length 5, 
        even though b has length 6.
    """

    len_a = len(a)
    len_b = len(b)

    best_score = 0        # The length of the longest contiguous match
    best_start = 0        # Where that contiguous match begins in 'a'
    best_end = 0          # Where that contiguous match ends in 'a' (non-inclusive)

    # Slide 'b' over every possible alignment in 'a'
    for start_idx in range(len_a - len_b + 1):
        local_score = 0   # Tracks current consecutive matches in this alignment
        # Compare characters a[start_idx + j] vs b[j]
        for j in range(len_b):
            if a[start_idx + j] == b[j]:
                # If we match, increase local consecutive match count
                local_score += 1
            else:
                # Mismatch ends the current consecutive run
                if local_score > best_score:
                    best_score = local_score
                    # The match ran from (start_idx + j - local_score) up to (start_idx + j)
                    best_start = start_idx + j - local_score
                    best_end = start_idx + j
                local_score = 0
        
        # If we ended on a match (local_score > 0), check one last time
        if local_score > best_score:
            best_score = local_score
            best_start = start_idx + len_b - local_score
            best_end = start_idx + len_b

    return best_start, best_end


def train_test_split(data: list, train_size: int) -> tuple[list, list]:
    random.shuffle(data)
    return data[:train_size], data[train_size:]
