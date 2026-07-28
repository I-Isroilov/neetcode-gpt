from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        
        tokens = list(corpus)
        merges: List[List[str]] = []

        for _ in range(num_merges):
            pair_counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break


            max_count = max(pair_counts.values())
            best_candidates = [pair for pair, count in pair_counts.items() if count == max_count]
            best_pair = min(best_candidates)

            merges.append(list(best_pair))

            merged_token = best_pair[0] + best_pair[1]
            new_tokens = []

            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return merges

    
    
