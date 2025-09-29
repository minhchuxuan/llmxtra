import google.generativeai as genai
import torch
import numpy as np
from collections import Counter, defaultdict
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
    
    def create_refinement_prompt(self, topic_words_en: List[str], topic_words_cn: List[str]) -> str:
        """
        Create prompt for refining all topics at once

        Args:
            topic_words_en: List of English topic word strings (each with 15 top words)
            topic_words_cn: List of Chinese topic word strings (each with 15 top words)

        Returns:
            Formatted prompt string for all topics
        """
        num_topics = len(topic_words_en)

        prompt = f"""Given the following cross-lingual topic words from English and Chinese for {num_topics} topics, please refine and improve each topic by:

1. For each topic, we provide the top 15 most probable words from each language.
2. Identify the main theme that connects these words across both languages for each topic
3. Remove any irrelevant or noisy words that don't fit the coherent theme
4. Add relevant words that strengthen the topic coherence and cross-lingual representation
5. Return exactly 20 words per language for each refined topic

IMPORTANT: Use only SINGLE WORDS, not compound words or phrases. Each word should be a standalone term.
Examples: 
- Good: "economy", "business", "market", "trade"
- Bad: "business_model", "stock_market", "trade-off", "economic policy"

"""

        # Add all topics to the prompt
        for k in range(num_topics):
            top_15_en = topic_words_en[k].split()
            top_15_cn = topic_words_cn[k].split()

            words_en_str = ", ".join(top_15_en)
            words_cn_str = ", ".join(top_15_cn)

            prompt += f"""
Topic {k}:
English top 15 words: {words_en_str}
Chinese top 15 words: {words_cn_str}
"""

        prompt += f"""

Please provide your response in a SIMPLE plain-text format (no JSON, no code block) for ALL {num_topics} topics, exactly as follows per topic:

Topic <id>: <brief theme>
EN: word1 - word2 - ... - word20
CN: word1 - word2 - ... - word20

Rules:
- Only use single words (no compound words, phrases, or underscores)
- Exactly 20 words after EN: and exactly 20 words after CN:
- Separate words with a hyphen surrounded by single spaces (e.g., "word1 - word2")
- List topics in order from 0 to {num_topics - 1}
- Do not include any extra commentary or formatting

Focus on the most coherent and representative single words from both languages for each topic.
"""
        return prompt
    
    def _parse_plain_response(self, response_text: str, expected_num_topics: int) -> List[Dict]:
        """Parse plain-text Topic/EN/CN response into a list of topic dicts."""
        topics = []
        # Split by lines and iterate assembling blocks per topic
        lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]
        i = 0
        while i < len(lines):
            # Expect: Topic k: theme
            m = re.match(r"^Topic\s+(\d+)\s*:\s*(.*)$", lines[i])
            if not m:
                i += 1
                continue
            topic_id = int(m.group(1))
            theme = m.group(2).strip()
            en_words = []
            cn_words = []
            if i + 1 < len(lines) and lines[i+1].startswith("EN:"):
                en_line = lines[i+1][3:].strip()
                # Support both hyphen-separated and comma-separated
                if ' - ' in en_line:
                    en_words = [w.strip() for w in en_line.split(' - ') if w.strip()]
                else:
                    en_words = [w.strip() for w in en_line.split(',') if w.strip()]
            if i + 2 < len(lines) and lines[i+2].startswith("CN:"):
                cn_line = lines[i+2][3:].strip()
                if ' - ' in cn_line:
                    cn_words = [w.strip() for w in cn_line.split(' - ') if w.strip()]
                else:
                    cn_words = [w.strip() for w in cn_line.split(',') if w.strip()]
            if en_words and cn_words:
                topics.append({
                    'topic_id': topic_id,
                    'topic_theme': theme,
                    'refined_words_en': en_words,
                    'refined_words_cn': cn_words
                })
                i += 3
            else:
                i += 1
        # Basic validation
        topics = sorted(topics, key=lambda t: t['topic_id'])
        topics = [t for t in topics if 0 <= t['topic_id'] < expected_num_topics]
        return topics if topics else None

    def _check_word_counts(self, topics: List[Dict], expected_count: int = 20) -> bool:
        """Return True if each topic has at least expected_count EN and CN words."""
        if not topics:
            return False
        ok = True
        for t in topics:
            en = t.get('refined_words_en', [])
            cn = t.get('refined_words_cn', [])
            if len(en) < expected_count or len(cn) < expected_count:
                tid = t.get('topic_id')
                print(f"Format check failed for topic {tid}: EN={len(en)}, CN={len(cn)} (expected at least {expected_count}).")
                ok = False
        return ok

    def call_gemini_api(self, prompt: str, expected_num_topics: int, max_retries: int = 3) -> List[Dict]:
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
                
                # Extract plain text from response and parse
                response_text = response.text
                # Parse using the expected number of topics provided by caller
                parsed = self._parse_plain_response(response_text, expected_num_topics=expected_num_topics)
                if parsed:
                    return parsed
                else:
                    preview = response_text.strip().splitlines()
                    preview_text = "\n".join(preview[:10])
                    print("Failed to parse LLM response. First lines preview:\n" + preview_text)
                        
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
        Self-Consistent Refinement: Ask Gemini R times to refine topics, count word occurrences

        Args:
            topic_words_en: List of English topic word strings (each with 15 words)
            topic_words_cn: List of Chinese topic word strings (each with 15 words)
            R: Number of refinement rounds

        Returns:
            List of topics with word counts across refinement rounds
        """
        num_topics = len(topic_words_en)
        
        # Initialize topic data structures
        refined_topics = []
        for k in range(num_topics):
            refined_topics.append({
                'topic_id': k,
                'word_counts_en': defaultdict(int),
                'word_counts_cn': defaultdict(int),
                'refinement_rounds_completed': 0
            })
        
        print(f"Starting refinement for {num_topics} topics with {R} rounds...")
        
        # Perform R refinement rounds
        for r in range(R):
            prompt = self.create_refinement_prompt(topic_words_en, topic_words_cn)
            result = self.call_gemini_api(prompt, expected_num_topics=num_topics)
            
            if not (result and isinstance(result, list)):
                print(f"Round {r+1}: Failed to get valid API results")
                continue
                
            # Process refinement results
            for topic_result in result:
                if not self._is_valid_topic_result(topic_result, num_topics):
                    continue
                    
                topic_id = topic_result['topic_id']
                topic_data = refined_topics[topic_id]
                
                # Update word counts for both languages
                self._update_word_counts(
                    topic_data['word_counts_en'], 
                    topic_result.get('refined_words_en', [])
                )
                self._update_word_counts(
                    topic_data['word_counts_cn'], 
                    topic_result.get('refined_words_cn', [])
                )
                
                # Track completed rounds
                topic_data['refinement_rounds_completed'] += 1
        
        return refined_topics
    
    def _is_valid_topic_result(self, topic_result: Dict, num_topics: int) -> bool:
        """Validate topic result structure"""
        return (isinstance(topic_result, dict) and 
                'topic_id' in topic_result and 
                topic_result['topic_id'] < num_topics)
    
    def _update_word_counts(self, word_counts: defaultdict, words: List[str]) -> None:
        """Update word counts efficiently"""
        for word in words:
            word_counts[word] += 1
    
    
    def get_high_confidence_words(self, 
                                  refined_topics: List[Dict], 
                                  top_k: int = 15) -> List[Dict]:
        """
        Get top-k words by count across refinement rounds
        
        Args:
            refined_topics: List of refined topic dictionaries with word counts
            top_k: Number of top words to return per topic (default 15)
            
        Returns:
            List with top words and their raw counts
        """
        results = []
        
        for topic_data in refined_topics:
            en_word_counts = topic_data.get('word_counts_en', {})
            cn_word_counts = topic_data.get('word_counts_cn', {})
            
            # Get top_k words by count (highest first)
            en_top_items = sorted(en_word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            cn_top_items = sorted(cn_word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            results.append({
                'topic_id': topic_data['topic_id'],
                'high_confidence_words_en': [word for word, count in en_top_items],
                'high_confidence_words_cn': [word for word, count in cn_top_items],
                'word_counts_en': {word: count for word, count in en_top_items},
                'word_counts_cn': {word: count for word, count in cn_top_items}
            })
        
        return results
    
    def calculate_confidence_word_probabilities(self, high_confidence_topics: List[Dict]) -> List[Dict]:
        """
        Calculate probabilities for high confidence words from their counts
        
        Args:
            high_confidence_topics: Topics with high confidence words and their counts
            
        Returns:
            Topics with added probability distributions for high confidence words
        """
        topics_with_probs = []
        
        for topic_data in high_confidence_topics:
            topic_with_probs = topic_data.copy()
            
            # Calculate probabilities for English high confidence words
            en_counts = topic_data.get('word_counts_en', {})
            en_total = sum(en_counts.values())
            if en_total > 0:
                topic_with_probs['word_probs_en'] = {
                    word: count / en_total for word, count in en_counts.items()
                }
            else:
                topic_with_probs['word_probs_en'] = {}
            
            # Calculate probabilities for Chinese high confidence words  
            cn_counts = topic_data.get('word_counts_cn', {})
            cn_total = sum(cn_counts.values())
            if cn_total > 0:
                topic_with_probs['word_probs_cn'] = {
                    word: count / cn_total for word, count in cn_counts.items()
                }
            else:
                topic_with_probs['word_probs_cn'] = {}
                
            topics_with_probs.append(topic_with_probs)
            
        return topics_with_probs

    def create_uniqueness_prompt(self, topic_words_en: List[str], topic_words_cn: List[str]) -> str:
        """
        Create prompt for Phase 3: ensuring word uniqueness across topics
        """
        num_topics = len(topic_words_en)

        prompt = f"""You are a cross-lingual topic modeling expert. Given {num_topics} topics with their high-confidence words, ensure WORD UNIQUENESS across topics while preserving semantic coherence and cross-lingual alignment.

PROCESS:
1. Identify duplicate words across topics.
2. Keep each duplicate in the most semantically relevant topic.
3. Replace duplicates in other topics with suitable synonyms.
4. Ensure replacements do not introduce new duplicates.
5. Verify each topic has exactly 15 unique words per language.
6. Maintain cross-lingual coherence between EN and CN word pairs.

REQUIREMENTS:
- Each word appears in ONLY ONE topic overall.
- Words must fit their topic semantically.
- Exactly 15 DIFFERENT single words per language per topic.
- Use only single words (no compounds/phrases).
- No internal duplicates within a topic.
- Cross-lingual coherence: EN and CN words should represent the same semantic concepts.
- Maintain thematic consistency between languages within each topic.
"""

        # Add current topic information (GIỐNG PHASE 2)
        for k in range(num_topics):
            en_words = topic_words_en[k].split()[:15]
            cn_words = topic_words_cn[k].split()[:15]

            en_words_str = ", ".join(en_words)
            cn_words_str = ", ".join(cn_words)

            prompt += f"""
Topic {k}:
EN: {en_words_str}
CN: {cn_words_str}
"""

        prompt += f"""

Provide your response in this format for ALL {num_topics} topics:

Topic <id>: <brief theme>  
EN: word1 - word2 - ... - word15  
CN: word1 - word2 - ... - word15

FINAL RULES:
- Words must be unique across all topics.
- Keep duplicates in the most relevant topic.
- Replace others with synonyms that do not conflict.
- Maintain coherence and semantic meaning.
- Exactly 15 unique words per language per topic.
- Cross-lingual coherence: EN and CN words must represent the same semantic concepts.
- Maintain thematic consistency between languages within each topic.
- Separate words with " - ".
- List topics in order from 0 to {num_topics - 1}.
- No extra commentary.
"""
        return prompt
    def self_consistent_uniqueness(self, topic_words_en: List[str], topic_words_cn: List[str], R: int = 3) -> List[Dict]:
        """
        Self-Consistent Uniqueness: Ask Gemini R times to ensure word uniqueness, count word occurrences
        (IDENTICAL TO PHASE 2 EXCEPT PROMPT)
        
        Args:
            topic_words_en: List of English topic word strings (each with 15 words)
            topic_words_cn: List of Chinese topic word strings (each with 15 words)
            R: Number of uniqueness rounds
            
        Returns:
            List of topics with word counts across uniqueness rounds
        """
        num_topics = len(topic_words_en)
        
        # Initialize topic data structures (GIỐNG PHASE 2)
        unique_topics = []
        for k in range(num_topics):
            unique_topics.append({
                'topic_id': k,
                'word_counts_en': defaultdict(int),
                'word_counts_cn': defaultdict(int),
                'refinement_rounds_completed': 0
            })
        
        print(f"Starting uniqueness refinement for {num_topics} topics with {R} rounds...")
        
        # Perform R uniqueness rounds
        for r in range(R):
            print(f"Phase 3 Uniqueness Round {r + 1}/{R}...")
            
            # Create prompt for this round (GIỐNG PHASE 2)
            prompt = self.create_uniqueness_prompt(topic_words_en, topic_words_cn)
            
            # Call API (GIỐNG PHASE 2)
            result = self.call_gemini_api(prompt, expected_num_topics=num_topics)
            
            if not (result and isinstance(result, list)):
                print(f"Round {r+1}: Failed to get valid API results")
                continue
                
            # Process refinement results (GIỐNG PHASE 2)
            for topic_result in result:
                if not self._is_valid_topic_result(topic_result, num_topics):
                    continue
                    
                topic_id = topic_result['topic_id']
                topic_data = unique_topics[topic_id]
                
                # Update word counts for both languages (GIỐNG PHASE 2)
                self._update_word_counts(
                    topic_data['word_counts_en'], 
                    topic_result.get('refined_words_en', [])
                )
                self._update_word_counts(
                    topic_data['word_counts_cn'], 
                    topic_result.get('refined_words_cn', [])
                )
                
                topic_data['refinement_rounds_completed'] += 1
        
        return unique_topics

    def ensure_word_uniqueness(self, high_confidence_topics: List[Dict], max_retries: int = 3) -> List[Dict]:
        """
        Phase 3: Ensure word uniqueness across topics using API
        
        Args:
            high_confidence_topics: Topics with high confidence words
            max_retries: Maximum number of API retries
            
        Returns:
            Topics with unique words across all topics
        """
        print(f"Phase 3: Ensuring word uniqueness across {len(high_confidence_topics)} topics...")
        
        for attempt in range(max_retries):
            try:
                prompt = self.create_uniqueness_prompt(high_confidence_topics)
                response = self.model.generate_content(prompt)
                response_text = response.text
                
                # Parse the response
                unique_topics = self._parse_plain_response(response_text, len(high_confidence_topics))
                
                if unique_topics:
                    print(f"Phase 3 completed successfully!")
                    return unique_topics
                else:
                    print(f"Attempt {attempt + 1}: Failed to parse uniqueness response")
                    
            except Exception as e:
                print(f"Uniqueness API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        print("Phase 3 failed after all retries. Returning original topics.")
        return high_confidence_topics


    def validate_words_against_vocab(self, refined_topics: List[Dict], vocab_en: List[str], vocab_cn: List[str]) -> List[Dict]:
        """
        Validate refined words against actual vocabulary files and discard invalid words
        
        Args:
            refined_topics: List of refined topic dictionaries
            vocab_en: English vocabulary list from TextData
            vocab_cn: Chinese vocabulary list from TextData
            
        Returns:
            List of validated refined topics with only vocab-valid words
        """
        vocab_en_set = set(vocab_en)
        vocab_cn_set = set(vocab_cn)
        
        validated_topics = []
        
        for topic_data in refined_topics:
            topic_id = topic_data['topic_id']
            
            # Get refined word counts
            word_counts_en = topic_data.get('word_counts_en', {})
            word_counts_cn = topic_data.get('word_counts_cn', {})
            
            # Filter words that exist in vocabulary
            valid_word_counts_en = {word: count for word, count in word_counts_en.items() 
                                   if word in vocab_en_set}
            valid_word_counts_cn = {word: count for word, count in word_counts_cn.items() 
                                   if word in vocab_cn_set}
            
            # Count discarded words for logging
            discarded_en = len(word_counts_en) - len(valid_word_counts_en)
            discarded_cn = len(word_counts_cn) - len(valid_word_counts_cn)
            
            # Create validated topic data
            validated_topic = topic_data.copy()
            validated_topic['word_counts_en'] = valid_word_counts_en
            validated_topic['word_counts_cn'] = valid_word_counts_cn
            validated_topic['discarded_words_en'] = discarded_en
            validated_topic['discarded_words_cn'] = discarded_cn
            
            validated_topics.append(validated_topic)
        
        return validated_topics


def refine_cross_lingual_topics(topic_words_en: List[str],
                                topic_words_cn: List[str], 
                                topic_probas_en: torch.Tensor,
                                topic_probas_cn: torch.Tensor,
                                vocab_en: List[str],
                                vocab_cn: List[str],
                                api_key: str,
                                R: int = 3,
                                enable_phase3: bool = True,
                                run_phase3: bool = False,
                                skip_phase2: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Main function to perform cross-lingual topic refinement for all topics at once

    Mathematical Framework:
    1. Process all topics simultaneously in each refinement round
    2. Self-consistent refinement: Refine top 15 words by removing irrelevant and adding relevant words, repeat R times
    3. Vocabulary validation: Discard refined words not in actual vocabulary files
    4. Frequency-based confidence: Aggregate across rounds for each topic
    5. Phase 3: Ensure word uniqueness across topics (replace duplicates with synonyms)

    Args:
        topic_words_en: English topic words (each with top 15 words)
        topic_words_cn: Chinese topic words (each with top 15 words)
        topic_probas_en: English topic probabilities [num_topics, 15] (top 15 words)
        topic_probas_cn: Chinese topic probabilities [num_topics, 15] (top 15 words)
        vocab_en: English vocabulary list from TextData
        vocab_cn: Chinese vocabulary list from TextData
        api_key: Gemini API key
        R: Number of refinement rounds
        enable_phase3: Whether to enable Phase 3 (word uniqueness across topics)
        skip_phase2: Skip Phase 2 (coherence refinement), only run Phase 3 (uniqueness refinement)

    Returns:
        Tuple of (refined_topics, high_confidence_topics)
    """
    refiner = CrossLingualTopicRefiner(api_key)
    
    if skip_phase2 and run_phase3:
        print("Skipping Phase 2 (coherence refinement), running Phase 3 (uniqueness refinement) on original topic words...")
        
        # Convert original topic words to the expected format for Phase 3
        original_topics = []
        for i in range(len(topic_words_en)):
            en_words = topic_words_en[i].split()[:15]
            cn_words = topic_words_cn[i].split()[:15]
            original_topics.append({
                'topic_id': i,
                'high_confidence_words_en': en_words,
                'high_confidence_words_cn': cn_words,
                'word_counts_en': {},
                'word_counts_cn': {},
                'word_probs_en': {},
                'word_probs_cn': {}
            })
        
        high_confidence_topics_with_probs = original_topics
        validated_topics = []  # Empty for consistency
        
    else:
        print(f"Starting batch refinement for {len(topic_words_en)} topics with {R} rounds each...")
        
        # Phase 2: Process all topics together in each refinement round (coherence refinement)
        print("Phase 2: Self-consistent coherence refinement...")
        refined_topics = refiner.self_consistent_refinement(topic_words_en, topic_words_cn, R=R)
        
        # Validate refined words against actual vocabulary
        print("Phase 2: Validating refined words against vocabulary...")
        validated_topics = refiner.validate_words_against_vocab(refined_topics, vocab_en, vocab_cn)
        
        # Extract high-confidence words based on frequency from validated topics
        high_confidence_topics = refiner.get_high_confidence_words(
            validated_topics, top_k=15
        )
        
        # Calculate probabilities for high confidence words
        print("Phase 2: Calculating probabilities for high confidence words...")
        high_confidence_topics_with_probs = refiner.calculate_confidence_word_probabilities(high_confidence_topics)
    
    # Phase 3: Ensure word uniqueness across topics (if enabled and requested)
    if enable_phase3 and run_phase3:
        print("Phase 3: Ensuring word uniqueness across topics with self-consistency...")
        
        # Phase 3 Self-consistent refinement (GIỐNG PHASE 2)
        print("Phase 3: Self-consistent uniqueness refinement...")
        unique_topics_with_counts = refiner.self_consistent_uniqueness(topic_words_en, topic_words_cn, R=R)
        
        # Phase 3 Vocabulary validation - convert to proper format first (GIỐNG PHASE 2)
        print("Phase 3: Validating unique words against vocabulary...")
        
        # Validate unique words against actual vocabulary (GIỐNG PHASE 2)
        validated_unique_topics = refiner.validate_words_against_vocab(unique_topics_with_counts, vocab_en, vocab_cn)
        
        # Extract high-confidence words based on frequency from validated topics (GIỐNG PHASE 2)
        print("Phase 3: Extracting high-confidence words...")
        high_confidence_topics_phase3 = refiner.get_high_confidence_words(
            validated_unique_topics, top_k=15
        )
        
        # Calculate probabilities for high confidence words (GIỐNG PHASE 2)
        print("Phase 3: Calculating probabilities for high confidence words...")
        final_high_confidence_topics_with_probs = refiner.calculate_confidence_word_probabilities(high_confidence_topics_phase3)
        
        return validated_topics, final_high_confidence_topics_with_probs
    else:
        if not enable_phase3:
            print("Phase 3 disabled. Returning Phase 2 results.")
        else:
            print("Phase 3 skipped. Returning Phase 2 results.")
        return validated_topics, high_confidence_topics_with_probs