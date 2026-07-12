from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # Start with characters as tokens
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # Count adjacent pairs
            pair_counts = Counter()
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += 1

            if not pair_counts:
                break

            # Most frequent pair; break ties lexicographically
            best_pair = min(
                pair_counts.items(),
                key=lambda x: (-x[1], x[0])
            )[0]

            merges.append([best_pair[0], best_pair[1]])

            # Merge all non-overlapping occurrences
            merged = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    merged.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        return merges