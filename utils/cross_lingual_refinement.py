import torch
import numpy as np
from collections import Counter, defaultdict
import re
import time
from typing import List, Dict, Tuple, Union
import warnings

warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")


class CrossLingualTopicRefiner:
    def __init__(self, api_key: str = None, model_name: str = "Qwen/Qwen2.5-32B-Instruct"):
        """
        Initialize the cross-lingual topic refiner with local Qwen model
        
        Args:
            api_key: Ignored (kept for compatibility with existing code)
            model_name: Path to Qwen model (default: Qwen/Qwen2.5-32B-Instruct)
        """
        print("Using local Qwen model - no API key required!")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
            
        print(f"Loading Qwen model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        print("Qwen model loaded successfully!")
    
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
5. Across ALL topics, do not reuse any word. If a word could fit multiple topics, assign it to the single best topic and replace it elsewhere.
6. For each topic, prioritize words that are specific to the theme and avoid words already used in other topics.
7. Return exactly 20 words per language for each refined topic

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
            
            # Look for EN: and CN: lines
            i += 1
            while i < len(lines):
                line = lines[i]
                if re.match(r"^Topic\s+\d+\s*:", line):
                    break
                if line.startswith("EN:"):
                    en_words = [w.strip() for w in line[3:].split("-") if w.strip()]
                elif line.startswith("CN:"):
                    cn_words = [w.strip() for w in line[3:].split("-") if w.strip()]
                i += 1
            
            topics.append({
                'topic_id': topic_id,
                'theme': theme,
                'refined_words_en': en_words,
                'refined_words_cn': cn_words
            })
        
        return topics
    
    def call_gemini_api(self, prompt: str, expected_num_topics: int, max_retries: int = 3) -> Union[List[Dict], None]:
        """
        Call Qwen model with retry logic (renamed for compatibility)
        
        Args:
            prompt: Input prompt for topic refinement
            expected_num_topics: Expected number of topics in response
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of topic dictionaries or None if failed
        """
        for attempt in range(max_retries):
            try:
                # Tokenize input
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                # Generate response
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=2048,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Parse response
                topics = self._parse_plain_response(response_text, expected_num_topics)
                
                if topics and len(topics) == expected_num_topics:
                    return topics
                else:
                    print(f"Attempt {attempt + 1}: Got {len(topics) if topics else 0} topics, expected {expected_num_topics}")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return None
    
    def self_consistent_refinement(self, 
                                  topic_words_en: List[str], 
                                  topic_words_cn: List[str], 
                                  R: int = 3) -> List[Dict]:
        """
        Perform self-consistent refinement across multiple rounds
        
        Args:
            topic_words_en: List of English topic word strings (each with 15 top words)
            topic_words_cn: List of Chinese topic word strings (each with 15 top words)
            R: Number of refinement rounds
            
        Returns:
            List of refined topic dictionaries with word counts
        """
        num_topics = len(topic_words_en)
        refined_topics = []
        
        # Initialize topic data structures
        for k in range(num_topics):
            refined_topics.append({
                'topic_id': k,
                'word_counts_en': defaultdict(int),
                'word_counts_cn': defaultdict(int),
                'refinement_rounds_completed': 0
            })
        
        print(f"Starting refinement for {num_topics} topics with {R} rounds...")
        
        # Perform R refinement rounds (process topics in 2 batches per round)
        for r in range(R):
            mid = (num_topics + 1) // 2
            batch_ranges = [(0, mid), (mid, num_topics)]
            batches_processed = 0

            for start, end in batch_ranges:
                if start >= end:
                    continue
                # Slice topics for this batch
                tw_en_batch = topic_words_en[start:end]
                tw_cn_batch = topic_words_cn[start:end]

                prompt = self.create_refinement_prompt(tw_en_batch, tw_cn_batch)
                result = self.call_gemini_api(prompt, expected_num_topics=len(tw_en_batch))

                if not (result and isinstance(result, list)):
                    print(f"Round {r+1}, batch [{start}:{end}]: Failed to get valid model results")
                    continue

                batches_processed += 1

                # Process refinement results for this batch (remap local ids to global ids)
                for topic_result in result:
                    local_tid = topic_result.get('topic_id')
                    if local_tid is None:
                        continue
                    global_tid = start + int(local_tid)
                    if not (0 <= global_tid < num_topics):
                        continue

                    topic_data = refined_topics[global_tid]

                    # Update word counts for both languages
                    self._update_word_counts(
                        topic_data['word_counts_en'], 
                        topic_result.get('refined_words_en', [])
                    )
                    self._update_word_counts(
                        topic_data['word_counts_cn'], 
                        topic_result.get('refined_words_cn', [])
                    )

                    # Track completed rounds per topic
                    topic_data['refinement_rounds_completed'] += 1

            print(f"Completed refinement round {r+1}/{R} (batches processed: {batches_processed}/2)")
        
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


def refine_cross_lingual_topics(topic_words_en: List[str], 
                               topic_words_cn: List[str], 
                               topic_probas_en: torch.Tensor,
                               topic_probas_cn: torch.Tensor,
                               vocab_en: List[str],
                               vocab_cn: List[str],
                               api_key: str = None,
                               R: int = 3) -> Tuple[List[Dict], List[Dict]]:
    """
    Main function to refine cross-lingual topics using local Qwen model
    
    Args:
        topic_words_en: List of English topic word strings
        topic_words_cn: List of Chinese topic word strings  
        topic_probas_en: English topic probability tensor
        topic_probas_cn: Chinese topic probability tensor
        vocab_en: English vocabulary
        vocab_cn: Chinese vocabulary
        api_key: Ignored (no API key needed for local model)
        R: Number of refinement rounds
        
    Returns:
        Tuple of (refined_topics, high_confidence_topics)
    """
    # Initialize refiner with local Qwen model (no API key needed)
    refiner = CrossLingualTopicRefiner(api_key=api_key)
    
    # Perform self-consistent refinement
    refined_topics = refiner.self_consistent_refinement(
        topic_words_en=topic_words_en,
        topic_words_cn=topic_words_cn,
        R=R
    )
    
    # Get high confidence words
    high_confidence_topics = refiner.get_high_confidence_words(
        refined_topics, 
        top_k=15
    )
    
    return refined_topics, high_confidence_topics