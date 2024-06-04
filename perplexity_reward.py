import numpy as np
import torch

from evaluate import load
from collections import Counter

from typing import List, Union

import evaluate
evaluate.logging.disable_progress_bar()


class PerplexityReward():
    def __init__(self, model_id, alpha=5.0, beta=0.001, ppl_threshold=(34.7 + 44.3)):
        self.model_id = model_id
        self.perplexity = load("perplexity", module_type="metric") # can be replaced with a custom implementation - using HF for now
        self.alpha = alpha
        self.beta = beta
        self.threshold = ppl_threshold
        
    def set_model(self, model_id):
        self.model_id = model_id
        self.perplexity = load("perplexity", module_type="metric")
        
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def set_alpha(self, alpha):
        self.alpha = alpha
        
    def set_beta(self, beta):
        self.beta = beta
        
    def get_ppl_stats(self, texts: Union[str, List[str]]) -> List[float]:
        """
        Get the perplexity of a list of texts.

        Args:
            texts (list): List of texts to compute the perplexity for.

        Returns:
            list: List of perplexities.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return self._get_perplexities(texts)
    
    def _get_word_freq_ratio(self, texts):
        """
        Get the ratio of the most frequent word to the total number of words in a text.

        Args:
            texts (list): List of texts to compute the ratio for.

        Returns:
            list: List of ratios. Returns 0 for texts that result in no words after processing.
        """
        
        freq_ratios = []
        for text in texts:
            words = text.split()
            words = ["".join(filter(str.isalnum, word)) for word in words]
            total_words = len(words)

            if total_words > 0:
                max_freq = max(Counter(words).values())
                freq_ratio = max_freq / total_words
            else:
                # didnt generate any sequence, hence give it full penalty
                freq_ratio = 1.0
            
            freq_ratios.append(freq_ratio)
        
        return freq_ratios

    
    def _compute_freq_reward(self, texts):
        freq_ratios = self._get_word_freq_ratio(texts)
        rewards = []
        
        for freq_ratio in freq_ratios:
            rewards.append(self.alpha * freq_ratio)
            
        return rewards
    
    
    def _get_perplexities(self, texts):
        return self.perplexity.compute(predictions=texts, model_id=self.model_id)["perplexities"]
    
    def _reward_fn(self, ppl):
        return self.alpha / (1 + np.exp(self.beta * (ppl - self.threshold)))
        
    def _compute_reward(self, texts):
        rewards = []
        perplexities = self._get_perplexities(texts)
        # ratio_offsets = self._compute_freq_reward(texts)
        
        for i in range(len(perplexities)):
            reward = self._reward_fn(perplexities[i])
            rewards.append(reward)  
            
        return rewards
    
    def __call__(self, texts, to_tensor=True):
        rewards = self._compute_reward(texts)
        
        if to_tensor:
            return [torch.tensor(reward) for reward in rewards]
        else:
            return rewards
