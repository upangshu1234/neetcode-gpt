from typing import List, Dict

class Solution:
    def _tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        """Helper method to perform greedy left-to-right longest match tokenization."""
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            match = None
            # Check substrings from longest possible to shortest (1 char)
            for j in range(n, i, -1):
                substring = text[i:j]
                if substring in vocab:
                    match = substring
                    i = j
                    break
            
            # Fallback: If no match is found in vocab, consume 1 character 
            # to prevent infinite loops (treating it as an unknown token).
            if match is None:
                match = text[i:i+1]
                i += 1
                
            tokens.append(match)
            
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        """Tokenize each number using greedy left-to-right longest match."""
        return [self._tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        """Count how many tokens the text uses with greedy tokenization."""
        return len(self._tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        """Compute tokens-per-word ratio (fertility)."""
        words = text.split()
        
        # Edge case: Empty text or text with only whitespaces
        if not words:
            return 0.0
            
        num_tokens = self.count_tokens(text, vocab)
        num_words = len(words)
        
        fertility = num_tokens / num_words
        return round(fertility, 4)