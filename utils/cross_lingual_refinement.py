import google.generativeai as genai
import torch
import numpy as np
from collections import Counter, defaultdict
import json
import re
import time
from typing import List, Dict, Tuple, Union


class CrossLingualTopicRefiner:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the cross-lingual topic refiner with Gemini API
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name to use
        """
    
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def cross_lingual_word_pooling(self, 
                                   topic_words_en: List[str], 
                                   topic_words_cn: List[str],
                                   topic_probas_en: torch.Tensor,
                                   topic_probas_cn: torch.Tensor,
                                   top_indices_en: torch.Tensor,
                                   top_indices_cn: torch.Tensor,
                                   vocab_en: List[str],
                                   vocab_cn: List[str]) -> List[Dict]:
        """
        Cross-Lingual Topic Word Pooling
        
        For each topic k, produces:
        W_k = w^{(A)}_k ∪ w^{(B)}_k (30 words total: 15 from each language)
        T_k = t^{(A)}_k ∪ t^{(B)}_k (corresponding probability distributions)
        
        Args:
            topic_words_en: List of English topic word strings (each has 15 words)
            topic_words_cn: List of Chinese topic word strings (each has 15 words)
            topic_probas_en: English topic probability distributions [num_topics, 15]
            topic_probas_cn: Chinese topic probability distributions [num_topics, 15]
            top_indices_en: English word indices [num_topics, 15]
            top_indices_cn: Chinese word indices [num_topics, 15]
            vocab_en: English vocabulary
            vocab_cn: Chinese vocabulary
            
        Returns:
            List of pooled topics with combined words and probabilities (30 words per topic)
        """
        pooled_topics = []
        num_topics = len(topic_words_en)
        
        for k in range(num_topics):
            # Extract words for topic k (15 words from each language)
            en_words = topic_words_en[k].split()  # Should be 15 words
            cn_words = topic_words_cn[k].split()  # Should be 15 words
            
            # Ensure we have exactly 15 words from each language
            en_words = en_words[:15] if len(en_words) >= 15 else en_words
            cn_words = cn_words[:15] if len(cn_words) >= 15 else cn_words
            
            # Get probabilities for topic k (15 probabilities from each language)
            en_probs = topic_probas_en[k].detach().cpu().numpy()[:15]
            cn_probs = topic_probas_cn[k].detach().cpu().numpy()[:15]
            
            # Create combined word set W_k = w^{(A)}_k ∪ w^{(B)}_k (30 words total)
            combined_words = en_words + cn_words  # 15 + 15 = 30 words
            combined_probs = np.concatenate([en_probs, cn_probs])  # 15 + 15 = 30 probabilities
            
            # Create topic-word distribution T_k = t^{(A)}_k ∪ t^{(B)}_k
            word_prob_dict = {}
            for word, prob in zip(combined_words, combined_probs):
                word_prob_dict[word] = float(prob)
            
            pooled_topic = {
                'topic_id': k,
                'words': combined_words,  # 30 words total
                'probabilities': combined_probs.tolist(),  # 30 probabilities
                'word_prob_dict': word_prob_dict,
                'en_words': en_words,  # 15 English words
                'cn_words': cn_words,  # 15 Chinese words
                'en_probs': en_probs.tolist(),  # 15 English probabilities
                'cn_probs': cn_probs.tolist(),  # 15 Chinese probabilities
                'total_words': len(combined_words)  # Should be 30
            }
            
            pooled_topics.append(pooled_topic)
            
        return pooled_topics
    
    def create_refinement_prompt(self, words: List[str], lang_a: str = "English", lang_b: str = "Chinese") -> str:
        """
        Create prompt for topic refinement
        
        Args:
            words: Combined list of words from both languages (30 words total: 15 from each language)
            lang_a: First language name
            lang_b: Second language name
            
        Returns:
            Formatted prompt string
        """
        words_str = ", ".join(words)
        
        prompt = f"""
Given the following cross-lingual topic words (15 words from {lang_a} and 15 words from {lang_b}, total 30 words), please refine and improve this topic by:

1. Identifying the main theme that connects these words across both languages
2. Removing any irrelevant or noisy words that don't fit the coherent theme
3. Adding relevant words that strengthen the topic coherence in both languages
4. Ensuring the refined word list maintains good cross-lingual representation

Original words (30 total): {words_str}

Please provide your response in the following JSON format:
{{
    "topic_theme": "brief description of the main topic theme",
    "refined_words": ["word1", "word2", "word3", ...],
    "removed_words": ["removed1", "removed2", ...],
    "added_words": ["added1", "added2", ...]
}}

Keep the refined word list exactly 30 words in total, focusing on the most coherent and representative words from both languages.
"""
        return prompt
    
    def call_gemini_api(self, prompt: str, max_retries: int = 3) -> Dict:
        """
        Call Gemini API with retry logic
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retries
            
        Returns:
            Parsed response dictionary
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                # Extract JSON from response
                response_text = response.text
                
                # Try to find JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    return result
                else:
                    print(f"No valid JSON found in response: {response_text}")
                    
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    
        return None
    
    def self_consistent_refinement(self, 
                                   pooled_topics: List[Dict], 
                                   R: int = 3,
                                   lang_a: str = "English", 
                                   lang_b: str = "Chinese") -> List[Dict]:
        """
        Self-Consistent Refinement
        
        Performs R refinements and aggregates results:
        W'_k = ⋃_{r=1}^R w^{(r)}_k
        
        Then creates normalized frequency distribution:
        Ũ_k(j) = Count(w_j ∈ W'_k) / Σ_{j'} Count(w_{j'} ∈ W'_k)
        
        Args:
            pooled_topics: List of pooled topic dictionaries (each with 30 words)
            R: Number of refinement rounds
            lang_a: First language name
            lang_b: Second language name
            
        Returns:
            List of refined topics with frequency-based confidence scores
        """
        refined_topics = []
        
        for topic_data in pooled_topics:
            topic_id = topic_data['topic_id']
            words = topic_data['words']  # 30 words (15 English + 15 Chinese)
            total_words = topic_data['total_words']  # Should be 30
            
            print(f"Refining topic {topic_id} with {R} rounds (starting with {total_words} words)...")
            
            # Collect refined words from R refinements
            all_refined_words = []  # Will collect words from all R refinement rounds
            refinement_details = []
            
            for r in range(R):
                print(f"  Refinement round {r+1}/{R}")
                
                prompt = self.create_refinement_prompt(words, lang_a, lang_b)
                result = self.call_gemini_api(prompt)
                
                if result and 'refined_words' in result:
                    refined_words_r = result['refined_words']  # w^{(r)}_k for round r
                    all_refined_words.extend(refined_words_r)
                    refinement_details.append(result)
                    print(f"    Got {len(refined_words_r)} refined words in round {r+1}")
                else:
                    print(f"    Failed to get valid response for round {r+1}")
            
            # Aggregate into pooled refined set W'_k = ⋃_{r=1}^R w^{(r)}_k
            word_counts = Counter(all_refined_words)
            
            # Create normalized frequency distribution Ũ_k(j)
            total_count = sum(word_counts.values())
            if total_count > 0:
                normalized_freq_dist = {
                    word: count / total_count 
                    for word, count in word_counts.items()
                }
            else:
                normalized_freq_dist = {}
            
            # Sort words by frequency (confidence) - higher frequency = higher confidence
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            
            refined_topic = {
                'topic_id': topic_id,
                'original_words': words,  # 30 original words
                'original_word_count': total_words,  # 30
                'refined_word_counts': dict(word_counts),  # Count(w_j ∈ W'_k)
                'normalized_freq_dist': normalized_freq_dist,  # Ũ_k(j)
                'sorted_refined_words': sorted_words,  # Sorted by frequency
                'refinement_details': refinement_details,
                'num_refinements': R,
                'total_refined_words': len(word_counts),
                'refinement_rounds_completed': len(refinement_details)
            }
            
            refined_topics.append(refined_topic)
            
        return refined_topics
    
    def get_high_confidence_words(self, 
                                  refined_topics: List[Dict], 
                                  min_frequency: float = 0.1,
                                  top_k: int = 10) -> List[Dict]:
        """
        Extract high-confidence words based on frequency threshold
        
        Args:
            refined_topics: List of refined topic dictionaries
            min_frequency: Minimum normalized frequency threshold
            top_k: Maximum number of words to return per topic
            
        Returns:
            List of high-confidence topic words
        """
        high_confidence_topics = []
        
        for topic_data in refined_topics:
            topic_id = topic_data['topic_id']
            freq_dist = topic_data['normalized_freq_dist']
            
            # Filter words by minimum frequency and take top-k
            high_conf_words = [
                (word, freq) for word, freq in freq_dist.items() 
                if freq >= min_frequency
            ]
            
            # Sort by frequency and take top-k
            high_conf_words.sort(key=lambda x: x[1], reverse=True)
            high_conf_words = high_conf_words[:top_k]
            
            high_confidence_topic = {
                'topic_id': topic_id,
                'high_confidence_words': [word for word, freq in high_conf_words],
                'word_frequencies': dict(high_conf_words),
                'num_high_conf_words': len(high_conf_words)
            }
            
            high_confidence_topics.append(high_confidence_topic)
            
        return high_confidence_topics


def refine_cross_lingual_topics(topic_words_en: List[str],
                                topic_words_cn: List[str], 
                                topic_probas_en: torch.Tensor,
                                topic_probas_cn: torch.Tensor,
                                top_indices_en: torch.Tensor,
                                top_indices_cn: torch.Tensor,
                                vocab_en: List[str],
                                vocab_cn: List[str],
                                api_key: str,
                                R: int = 3,
                                min_frequency: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """
    Main function to perform cross-lingual topic refinement
    
    Mathematical Framework:
    1. Cross-lingual pooling: W_k = w^{(A)}_k ∪ w^{(B)}_k (30 words: 15 English + 15 Chinese)
    2. Self-consistent refinement: W'_k = ⋃_{r=1}^R w^{(r)}_k
    3. Frequency-based confidence: Ũ_k(j) = Count(w_j ∈ W'_k) / Σ_{j'} Count(w_{j'} ∈ W'_k)
    
    Args:
        topic_words_en: English topic words (each topic has 15 words)
        topic_words_cn: Chinese topic words (each topic has 15 words)
        topic_probas_en: English topic probabilities [num_topics, 15]
        topic_probas_cn: Chinese topic probabilities [num_topics, 15]
        top_indices_en: English word indices [num_topics, 15]
        top_indices_cn: Chinese word indices [num_topics, 15]
        vocab_en: English vocabulary
        vocab_cn: Chinese vocabulary
        api_key: Gemini API key
        R: Number of refinement rounds
        min_frequency: Minimum frequency threshold for high-confidence words
        
    Returns:
        Tuple of (refined_topics, high_confidence_topics)
    """
    refiner = CrossLingualTopicRefiner(api_key)
    
    # Step 1: Cross-lingual word pooling - W_k = w^{(A)}_k ∪ w^{(B)}_k (30 words per topic)
    pooled_topics = refiner.cross_lingual_word_pooling(
        topic_words_en, topic_words_cn,
        topic_probas_en, topic_probas_cn,
        top_indices_en, top_indices_cn,
        vocab_en, vocab_cn
    )
    
    print(f"Created pooled topics: {len(pooled_topics)} topics, each with 30 words (15 EN + 15 CN)")
    
    # Step 2: Self-consistent refinement - W'_k = ⋃_{r=1}^R w^{(r)}_k
    refined_topics = refiner.self_consistent_refinement(pooled_topics, R=R)
    
    # Step 3: Extract high-confidence words based on frequency
    high_confidence_topics = refiner.get_high_confidence_words(
        refined_topics, min_frequency=min_frequency
    )
    
    return refined_topics, high_confidence_topics