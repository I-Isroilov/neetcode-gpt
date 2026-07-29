from typing import List, Dict

class Solution:
    def _greedy_tokenize(self, s:str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(s)
        max_len = max((len(tok) for tok in vocab), default=1)

        while i < n:
            matched = None

            for length in range(min(max_len, n - i), 0, -1):
                candidate = s[i:i + length]
                if candidate in vocab:
                    matched = candidate
                    break
            if matched is None:
                matched = s[i]
            tokens.append(matched)
            i += len(matched)
        return tokens
    
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self._greedy_tokenize(str(number), vocab) for number in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()
        if not words:
            return 0.0

        total_tokens = self.count_tokens(text, vocab)
        fertility = total_tokens / len(words)
        return round(fertility, 4)
        
