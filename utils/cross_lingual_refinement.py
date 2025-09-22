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
        Create prompt for ID-based topic refinement (closed-world selection)

        Args:
            topic_words_en: List of English topic word strings (each with 50 words vocabulary)
            topic_words_cn: List of Chinese topic word strings (each with 50 words vocabulary)

        Returns:
            Formatted prompt string for all topics with enumerated candidates
        """
        num_topics = len(topic_words_en)
        min_support_per_lang = 3

        header = f"""STRICT TOPIC REFINEMENT — ID SELECTION ONLY

You will process {num_topics} topics. For each topic:
- Identify a brief theme string.
- Select ≥{min_support_per_lang} EN IDs and ≥{min_support_per_lang} CN IDs.
- IDs MUST come only from the enumerated candidate lists.
- IDs must be integers, unique within each language, and in range.
- Output MUST be valid JSON, with no extra text, no markdown fences, no comments.

Return exactly one JSON object of the form:
{{
  "topics": [
    {{"topic_id": 0, "theme": "string", "selected_ids_en": [0,1,2], "selected_ids_cn": [0,1,2]}},
    {{"topic_id": 1, "theme": "string", "selected_ids_en": [0,1,2], "selected_ids_cn": [0,1,2]}}
  ]
}}
"""

        body = []
        for k in range(num_topics):
            en_list = topic_words_en[k].split()
            cn_list = topic_words_cn[k].split()

            en_block = " ".join(f"[{i}] {w}," for i, w in enumerate(en_list)).rstrip(",")
            cn_block = " ".join(f"[{i}] {w}," for i, w in enumerate(cn_list)).rstrip(",")

            body.append(
                f"\nTOPIC {k}\nEN_CANDIDATES:\n{en_block}\nCN_CANDIDATES:\n{cn_block}\n"
            )

        return header + "".join(body)
    
    def call_gemini_api(self, prompt: str, candidate_lists: List[Tuple[List[str], List[str]]], max_retries: int = 3) -> List[Dict]:
        """
        Call Gemini API with strict ID-based validation
        
        Args:
            prompt: Input prompt containing all topics
            candidate_lists: List of (en_candidates, cn_candidates) for validation
            max_retries: Maximum number of retries
            
        Returns:
            List of validated topic results with materialized words
        """
        min_support_per_lang = 3
        
        # Force JSON-only responses from Gemini
        gen_cfg = genai.types.GenerationConfig(
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=8000,
            response_mime_type="application/json",
        )
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt, generation_config=gen_cfg)
                response_text = response.text
                
                # Parse pure JSON response (guaranteed by response_mime_type)
                result = json.loads(response_text)
                
                if not isinstance(result, dict) or 'topics' not in result:
                    raise RuntimeError("Missing 'topics' key in response")
                    
                topics = result['topics']
                if not isinstance(topics, list):
                    raise RuntimeError("'topics' must be a list")
                
                validated_topics = []
                for topic_data in topics:
                    if not isinstance(topic_data, dict):
                        raise RuntimeError("Each topic must be a dict")
                    
                    # Extract and validate required fields
                    topic_id = topic_data.get('topic_id')
                    theme = topic_data.get('theme', '')
                    selected_ids_en = topic_data.get('selected_ids_en', [])
                    selected_ids_cn = topic_data.get('selected_ids_cn', [])
                    
                    if not isinstance(topic_id, int) or topic_id < 0 or topic_id >= len(candidate_lists):
                        raise RuntimeError(f"Invalid topic_id: {topic_id}")
                    
                    if not isinstance(selected_ids_en, list) or not isinstance(selected_ids_cn, list):
                        raise RuntimeError(f"selected_ids must be lists for topic {topic_id}")
                    
                    en_candidates, cn_candidates = candidate_lists[topic_id]
                    
                    # Validate EN IDs
                    if not all(isinstance(id_, int) for id_ in selected_ids_en):
                        raise RuntimeError(f"All EN IDs must be integers for topic {topic_id}")
                    if not all(0 <= id_ < len(en_candidates) for id_ in selected_ids_en):
                        raise RuntimeError(f"EN IDs out of range [0, {len(en_candidates)-1}] for topic {topic_id}")
                    if len(set(selected_ids_en)) != len(selected_ids_en):
                        raise RuntimeError(f"Duplicate EN IDs for topic {topic_id}")
                    if len(selected_ids_en) < min_support_per_lang:
                        raise RuntimeError(f"Too few EN words ({len(selected_ids_en)} < {min_support_per_lang}) for topic {topic_id}")
                    
                    # Validate CN IDs
                    if not all(isinstance(id_, int) for id_ in selected_ids_cn):
                        raise RuntimeError(f"All CN IDs must be integers for topic {topic_id}")
                    if not all(0 <= id_ < len(cn_candidates) for id_ in selected_ids_cn):
                        raise RuntimeError(f"CN IDs out of range [0, {len(cn_candidates)-1}] for topic {topic_id}")
                    if len(set(selected_ids_cn)) != len(selected_ids_cn):
                        raise RuntimeError(f"Duplicate CN IDs for topic {topic_id}")
                    if len(selected_ids_cn) < min_support_per_lang:
                        raise RuntimeError(f"Too few CN words ({len(selected_ids_cn)} < {min_support_per_lang}) for topic {topic_id}")
                    
                    # Materialize words by index lookup
                    ref_words_en = [en_candidates[i] for i in selected_ids_en]
                    ref_words_cn = [cn_candidates[i] for i in selected_ids_cn]
                    
                    validated_topics.append({
                        'topic_id': topic_id,
                        'theme': theme,
                        'refined_words_en': ref_words_en,  # Keep original field names for compatibility
                        'refined_words_cn': ref_words_cn
                    })
                
                return validated_topics
                        
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        raise RuntimeError("All API attempts failed with validation errors")
    
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
        
        # Build candidate lists for validation
        candidate_lists = []
        for k in range(num_topics):
            en_candidates = topic_words_en[k].split()
            cn_candidates = topic_words_cn[k].split()
            candidate_lists.append((en_candidates, cn_candidates))
        
        for r in range(R):
            print(f"Refinement round {r+1}/{R} for all topics...")
            
            # Create prompt for all topics
            prompt = self.create_refinement_prompt(topic_words_en, topic_words_cn)
            
            try:
                result = self.call_gemini_api(prompt, candidate_lists)
                if len(result) != num_topics:
                    raise RuntimeError(f"[Refine] Round {r+1}: got {len(result)} topics, expected {num_topics}.")
                print(f"Round {r+1}: Got validated results for {len(result)} topics")
                
                # Process validated results (words are already materialized and guaranteed in-vocab)
                for topic_result in result:
                    topic_id = topic_result['topic_id']
                    
                    # Extract refined words (already validated and materialized)
                    refined_words_en = topic_result['refined_words_en']
                    refined_words_cn = topic_result['refined_words_cn']
                    
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
                    
            except RuntimeError as e:
                print(f"Round {r+1}: Validation failed - {e}")
                raise RuntimeError(f"[Refine] Round {r+1} failed: {e}")
        
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
                                  top_k: int = 15) -> List[Dict]:
        """
        Extract top-k high-confidence words based on frequency ranking
        
        Args:
            refined_topics: List of refined topic dictionaries
            top_k: Maximum number of words to return per topic
            
        Returns:
            List of high-confidence topic words
        """
        high_confidence_topics = []
        
        for topic_data in refined_topics:
            topic_id = topic_data['topic_id']
            freq_dist_en = topic_data.get('normalized_freq_dist_en', {})
            freq_dist_cn = topic_data.get('normalized_freq_dist_cn', {})

            
            # For English - take top_k words by frequency
            high_conf_words_en = [
                (word, freq) for word, freq in freq_dist_en.items()
            ]
            high_conf_words_en.sort(key=lambda x: x[1], reverse=True)
            high_conf_words_en = high_conf_words_en[:top_k]
            
            # For Chinese - take top_k words by frequency
            high_conf_words_cn = [
                (word, freq) for word, freq in freq_dist_cn.items()
            ]
            high_conf_words_cn.sort(key=lambda x: x[1], reverse=True)
            high_conf_words_cn = high_conf_words_cn[:top_k]
            
            # Enforce minimum usable support per language per topic
            min_support_per_lang = 2  # must match the OT MIN_SUPPORT_PER_TOPIC
            en_words = [word for word, freq in high_conf_words_en]
            cn_words = [word for word, freq in high_conf_words_cn]
            
            if len(en_words) < min_support_per_lang and len(cn_words) < min_support_per_lang:
                raise RuntimeError(
                    f"[Refine] Topic {topic_id}: insufficient high-confidence words "
                    f"(EN={len(en_words)}, CN={len(cn_words)}; need ≥{min_support_per_lang} on at least one side)"
                )
            
            high_confidence_topic = {
                'topic_id': topic_id,
                'high_confidence_words_en': en_words,
                'high_confidence_words_cn': cn_words,
                'word_frequencies_en': dict(high_conf_words_en),
                'word_frequencies_cn': dict(high_conf_words_cn),
                'num_high_conf_words_en': len(en_words),
                'num_high_conf_words_cn': len(cn_words)
            }
            
            high_confidence_topics.append(high_confidence_topic)
        
        return high_confidence_topics


def refine_cross_lingual_topics(topic_words_en: List[str],
                                topic_words_cn: List[str], 
                                topic_probas_en: torch.Tensor,
                                topic_probas_cn: torch.Tensor,
                                api_key: str,
                                R: int = 3) -> Tuple[List[Dict], List[Dict]]:
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

    Returns:
        Tuple of (refined_topics, high_confidence_topics)
    """
    refiner = CrossLingualTopicRefiner(api_key)
    
    print(f"Starting batch refinement for {len(topic_words_en)} topics with {R} rounds each...")
    
    # Process all topics together in each refinement round
    refined_topics = refiner.self_consistent_refinement(topic_words_en, topic_words_cn, R=R)
    
    # Extract high-confidence words based on frequency
    high_confidence_topics = refiner.get_high_confidence_words(
        refined_topics, top_k=15
    )
    
    return refined_topics, high_confidence_topics