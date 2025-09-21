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
                                   topic_probas_cn: torch.Tensor) -> List[Dict]:
        """
        Cross-Lingual Topic Word Pooling

        For each topic k, produces:
        W_k = w^{(A)}_k ∪ w^{(B)}_k (combined words from both languages)
        T_k = t^{(A)}_k ∪ t^{(B)}_k (corresponding probability distributions)

        Args:
            topic_words_en: List of English topic word strings (each with 50 words vocabulary)
            topic_words_cn: List of Chinese topic word strings (each with 50 words vocabulary)
            topic_probas_en: English topic probability distributions [num_topics, 15]
            topic_probas_cn: Chinese topic probability distributions [num_topics, 15]

        Returns:
            List of pooled topics with combined words and probabilities
        """
        pooled_topics = []
        num_topics = len(topic_words_en)
        
        for k in range(num_topics):
            # Extract all 50 words for topic k (vocabulary)
            en_words_50 = topic_words_en[k].split()[:50]
            cn_words_50 = topic_words_cn[k].split()[:50]

            # Take top 15 words for refinement focus
            en_words = en_words_50[:15]
            cn_words = cn_words_50[:15]

            # Match probabilities to actual word count (top 15)
            en_probs = topic_probas_en[k].detach().cpu().numpy()[:len(en_words)]
            cn_probs = topic_probas_cn[k].detach().cpu().numpy()[:len(cn_words)]

            # Create combined word set and probabilities
            combined_words = en_words + cn_words
            combined_probs = np.concatenate([en_probs, cn_probs])
            
            # Handle duplicate words by accumulating probabilities
            word_prob_dict = {}
            for word, prob in zip(combined_words, combined_probs):
                word_prob_dict[word] = word_prob_dict.get(word, 0.0) + float(prob)
            
            pooled_topic = {
                'topic_id': k,
                'words': combined_words,
                'probabilities': combined_probs.tolist(),
                'word_prob_dict': word_prob_dict,
                'en_words': en_words,
                'cn_words': cn_words,
                'en_probs': en_probs.tolist(),
                'cn_probs': cn_probs.tolist(),
                'total_words': len(combined_words)
            }
            
            pooled_topics.append(pooled_topic)
            
        return pooled_topics
    
    def create_refinement_prompt(self, topic_words_en: List[str], topic_words_cn: List[str]) -> str:
        """
        Create prompt for refining all topics at once

        Args:
            topic_words_en: List of English topic word strings (each with 50 words vocabulary)
            topic_words_cn: List of Chinese topic word strings (each with 50 words vocabulary)

        Returns:
            Formatted prompt string for all topics
        """
        num_topics = len(topic_words_en)

        prompt = f"""Given the following cross-lingual topic words from English and Chinese for {num_topics} topics, please refine and improve each topic by:

1. For each topic, we provide a vocabulary of top 50 words from each language.
2. The top 15 words from each language are the most probable (first 15 in the list) and represent the current topic theme.
3. Identify the main theme that connects these top 15 words across both languages for each topic
4. Remove any irrelevant or noisy words from the top 15 that don't fit the coherent theme
5. Add relevant words from the full top 50 vocabulary list that strengthen the topic coherence
6. Ensure each refined topic has exactly 15 words per language that maintain good cross-lingual representation

"""

        # Add all topics to the prompt
        for k in range(num_topics):
            top_50_en = topic_words_en[k].split()
            top_50_cn = topic_words_cn[k].split()

            words_en_str = ", ".join(top_50_en)
            words_cn_str = ", ".join(top_50_cn)

            prompt += f"""
Topic {k}:
English top 50 words: {words_en_str}
Chinese top 50 words: {words_cn_str}
"""

        prompt += f"""

Please provide your response in the following JSON format for ALL {num_topics} topics:
{{
    "topics": [
        {{
            "topic_id": 0,
            "topic_theme": "brief description of topic 0 theme",
            "refined_words_en": ["word1", "word2", ..., "word15"],
            "refined_words_cn": ["word1", "word2", ..., "word15"],
            "removed_words": ["removed1", "removed2", ...],
            "added_words": ["added1", "added2", ...]
        }},
        {{
            "topic_id": 1,
            "topic_theme": "brief description of topic 1 theme",
            "refined_words_en": ["word1", "word2", ..., "word15"],
            "refined_words_cn": ["word1", "word2", ..., "word15"],
            "removed_words": ["removed1", "removed2", ...],
            "added_words": ["added1", "added2", ...]
        }},
        // ... continue for all {num_topics} topics
    ]
}}

Focus on the most coherent and representative words from both languages for each topic.
"""
        return prompt
    
    def call_gemini_api(self, prompt: str, max_retries: int = 3) -> List[Dict]:
        """
        Call Gemini API with retry logic for multiple topics
        
        Args:
            prompt: Input prompt containing all topics
            max_retries: Maximum number of retries
            
        Returns:
            List of parsed response dictionaries for each topic
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
                    
                    # Check if result has the expected structure
                    if 'topics' in result and isinstance(result['topics'], list):
                        return result['topics']
                    else:
                        print(f"Unexpected JSON structure: {result}")
                        
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    
        return None
    
    def self_consistent_refinement(self,
                                   topic_words_en: List[str],
                                   topic_words_cn: List[str],
                                   R: int = 3) -> List[Dict]:
        """
        Self-Consistent Refinement for all topics at once

        Performs R refinements for all topics in each round.

        Args:
            topic_words_en: List of English topic word strings (each with 50 words vocabulary)
            topic_words_cn: List of Chinese topic word strings (each with 50 words vocabulary)
            R: Number of refinement rounds

        Returns:
            List of refined topics with frequency-based confidence scores
        """
        num_topics = len(topic_words_en)
        refined_topics = []
        
        # Initialize collections for all topics
        for k in range(num_topics):
            refined_topics.append({
                'topic_id': k,
                'original_top_50_en': topic_words_en[k].split(),
                'original_top_50_cn': topic_words_cn[k].split(),
                'refined_word_counts_en': {},
                'refined_word_counts_cn': {},
                'normalized_freq_dist_en': {},
                'normalized_freq_dist_cn': {},
                'refinement_details': [],
                'num_refinements': R,
                'total_refined_words_en': 0,
                'total_refined_words_cn': 0,
                'refinement_rounds_completed': 0
            })
        
        print(f"Starting refinement for {num_topics} topics with {R} rounds...")
        
        for r in range(R):
            print(f"Refinement round {r+1}/{R} for all topics...")
            
            # Create prompt for all topics
            prompt = self.create_refinement_prompt(topic_words_en, topic_words_cn)
            result = self.call_gemini_api(prompt)
            
            print(f"Round {r+1}: API result type: {type(result)}, length: {len(result) if result else 'None'}")
            
            if result and isinstance(result, list):
                print(f"Round {r+1}: Got refinement results for {len(result)} topics")
                
                # Process results for each topic
                for topic_result in result:
                    if isinstance(topic_result, dict) and 'topic_id' in topic_result:
                        topic_id = topic_result['topic_id']
                        
                        if topic_id < num_topics:
                            # Extract refined words
                            refined_words_en = topic_result.get('refined_words_en', [])
                            refined_words_cn = topic_result.get('refined_words_cn', [])
                            
                            # Update word counts
                            for word in refined_words_en:
                                refined_topics[topic_id]['refined_word_counts_en'][word] = \
                                    refined_topics[topic_id]['refined_word_counts_en'].get(word, 0) + 1
                            
                            for word in refined_words_cn:
                                refined_topics[topic_id]['refined_word_counts_cn'][word] = \
                                    refined_topics[topic_id]['refined_word_counts_cn'].get(word, 0) + 1
                            
                            # Store refinement details
                            refined_topics[topic_id]['refinement_details'].append(topic_result)
                            refined_topics[topic_id]['refinement_rounds_completed'] += 1
            else:
                print(f"Round {r+1}: Failed to get valid results")
        
        # Calculate final statistics for each topic
        for k in range(num_topics):
            # English words
            word_counts_en = refined_topics[k]['refined_word_counts_en']
            total_count_en = sum(word_counts_en.values())
            if total_count_en > 0:
                refined_topics[k]['normalized_freq_dist_en'] = {
                    word: count / total_count_en 
                    for word, count in word_counts_en.items()
                }
            refined_topics[k]['total_refined_words_en'] = len(word_counts_en)
            
            # Chinese words
            word_counts_cn = refined_topics[k]['refined_word_counts_cn']
            total_count_cn = sum(word_counts_cn.values())
            if total_count_cn > 0:
                refined_topics[k]['normalized_freq_dist_cn'] = {
                    word: count / total_count_cn 
                    for word, count in word_counts_cn.items()
                }
            refined_topics[k]['total_refined_words_cn'] = len(word_counts_cn)
        
        return refined_topics
    
    def get_high_confidence_words(self, 
                                  refined_topics: List[Dict], 
                                  min_frequency: float = 0.1,
                                  top_k: int = 15) -> List[Dict]:
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
            freq_dist_en = topic_data.get('normalized_freq_dist_en', {})
            freq_dist_cn = topic_data.get('normalized_freq_dist_cn', {})

            
            # For English
            high_conf_words_en = [
                (word, freq) for word, freq in freq_dist_en.items() 
                if freq >= min_frequency
            ]
            high_conf_words_en.sort(key=lambda x: x[1], reverse=True)
            high_conf_words_en = high_conf_words_en[:top_k]
            
            # For Chinese
            high_conf_words_cn = [
                (word, freq) for word, freq in freq_dist_cn.items() 
                if freq >= min_frequency
            ]
            high_conf_words_cn.sort(key=lambda x: x[1], reverse=True)
            high_conf_words_cn = high_conf_words_cn[:top_k]
            
            high_confidence_topic = {
                'topic_id': topic_id,
                'high_confidence_words_en': [word for word, freq in high_conf_words_en],
                'high_confidence_words_cn': [word for word, freq in high_conf_words_cn],
                'word_frequencies_en': dict(high_conf_words_en),
                'word_frequencies_cn': dict(high_conf_words_cn),
                'num_high_conf_words_en': len(high_conf_words_en),
                'num_high_conf_words_cn': len(high_conf_words_cn)
            }
            
            high_confidence_topics.append(high_confidence_topic)
        
        return high_confidence_topics


def refine_cross_lingual_topics(topic_words_en: List[str],
                                topic_words_cn: List[str], 
                                topic_probas_en: torch.Tensor,
                                topic_probas_cn: torch.Tensor,
                                api_key: str,
                                R: int = 3,
                                min_frequency: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """
    Main function to perform cross-lingual topic refinement for all topics at once

    Mathematical Framework:
    1. Process all 50 topics simultaneously in each refinement round
    2. Self-consistent refinement: Refine top 15 words by removing irrelevant and adding from 50-word vocabulary, repeat R times
    3. Frequency-based confidence: Aggregate across rounds for each topic

    Args:
        topic_words_en: English topic words (50 topics, each with 50 words vocabulary)
        topic_words_cn: Chinese topic words (50 topics, each with 50 words vocabulary)
        topic_probas_en: English topic probabilities [50, 15] (top 15 words)
        topic_probas_cn: Chinese topic probabilities [50, 15] (top 15 words)
        api_key: Gemini API key
        R: Number of refinement rounds
        min_frequency: Minimum frequency threshold for high-confidence words

    Returns:
        Tuple of (refined_topics, high_confidence_topics)
    """
    refiner = CrossLingualTopicRefiner(api_key)
    
    print(f"Starting batch refinement for {len(topic_words_en)} topics with {R} rounds each...")
    
    # Process all topics together in each refinement round
    refined_topics = refiner.self_consistent_refinement(topic_words_en, topic_words_cn, R=R)
    
    # Extract high-confidence words based on frequency
    high_confidence_topics = refiner.get_high_confidence_words(
        refined_topics, min_frequency=min_frequency, top_k=15
    )
    
    return refined_topics, high_confidence_topics