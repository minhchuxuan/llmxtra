import torch
import numpy as np
import ot
from sentence_transformers import SentenceTransformer


def compute_cross_lingual_refine_loss(topic_probas_en, topic_probas_cn, 
                                      refined_topics, high_confidence_topics,
                                      vocab_en, vocab_cn, 
                                      embedding_model=None):
    """
    Compute refinement loss for cross-lingual topic modeling using proper OT.
    
    Args:
        topic_probas_en: English topic probabilities [num_topics, 15]
        topic_probas_cn: Chinese topic probabilities [num_topics, 15] 
        refined_topics: Output from cross-lingual refinement
        high_confidence_topics: High confidence words from refinement
        vocab_en: English vocabulary
        vocab_cn: Chinese vocabulary
        embedding_model: Sentence transformer model for cost computation
        
    Returns:
        refine_loss: Scalar tensor representing refinement loss
    """
    if embedding_model is None:
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    num_topics = topic_probas_en.shape[0]
    ot_dists = torch.zeros(size=(num_topics,)).cuda()
    
    for i in range(num_topics):
        if (i < len(high_confidence_topics) and 
            (high_confidence_topics[i]['high_confidence_words_en'] or 
             high_confidence_topics[i]['high_confidence_words_cn'])):
            # Get current topic words and probabilities  
            en_indices = torch.topk(topic_probas_en[i], k=15).indices
            en_words = [vocab_en[idx] for idx in en_indices.cpu().numpy()]
            en_probs = topic_probas_en[i][:len(en_words)].to(torch.float64)
            
            cn_indices = torch.topk(topic_probas_cn[i], k=15).indices 
            cn_words = [vocab_cn[idx] for idx in cn_indices.cpu().numpy()]
            cn_probs = topic_probas_cn[i][:len(cn_words)].to(torch.float64)
            
            # Create proper combined distribution (avoid double counting, normalize to sum=1)
            current_word_prob = {}
            lang_weight = 0.5  # Equal weight for both languages
            
            for word, prob in zip(en_words, en_probs):
                current_word_prob[word] = current_word_prob.get(word, 0) + prob.item() * lang_weight
            for word, prob in zip(cn_words, cn_probs):
                current_word_prob[word] = current_word_prob.get(word, 0) + prob.item() * lang_weight
            
            # Get refined words and normalize frequencies to probabilities
            refined_words_en = high_confidence_topics[i]['high_confidence_words_en']
            refined_words_cn = high_confidence_topics[i]['high_confidence_words_cn']
            freq_dict_en = high_confidence_topics[i]['word_frequencies_en']
            freq_dict_cn = high_confidence_topics[i]['word_frequencies_cn']
            
            # Normalize frequencies to create proper probability distribution
            total_freq = sum(freq_dict_en.values()) + sum(freq_dict_cn.values())
            refined_word_prob = {}
            
            if total_freq > 0:
                for word, freq in freq_dict_en.items():
                    refined_word_prob[word] = refined_word_prob.get(word, 0) + (freq / total_freq) * lang_weight
                for word, freq in freq_dict_cn.items():
                    refined_word_prob[word] = refined_word_prob.get(word, 0) + (freq / total_freq) * lang_weight
            
            if len(refined_word_prob) > 0:
                # Create union vocabulary for proper alignment
                all_words = list(set(current_word_prob.keys()) | set(refined_word_prob.keys()))
                vocab_size = len(all_words)
                
                # Create aligned distributions
                current_aligned = torch.zeros(vocab_size, dtype=torch.float64).cuda()
                refined_aligned = torch.zeros(vocab_size, dtype=torch.float64).cuda()
                
                # Map distributions to aligned vectors
                for idx, word in enumerate(all_words):
                    current_aligned[idx] = current_word_prob.get(word, 0.0)
                    refined_aligned[idx] = refined_word_prob.get(word, 0.0)
                
                # Ensure proper normalization (they should already sum to ~1, but numerical safety)
                if current_aligned.sum() > 0:
                    current_aligned = current_aligned / current_aligned.sum()
                if refined_aligned.sum() > 0:
                    refined_aligned = refined_aligned / refined_aligned.sum()
                
                # Compute cost matrix and OT distance
                cost_M = compute_embedding_cost_matrix(all_words, all_words, embedding_model)
                dist = ot.emd2(current_aligned, refined_aligned, cost_M)
                ot_dists[i] = dist
            else:
                ot_dists[i] = 0.0
        else:
            ot_dists[i] = 0.0
    
    # Compute topic weights based on refinement confidence
    topic_weights = torch.zeros(size=(num_topics,)).cuda()
    for i in range(num_topics):
        if i < len(refined_topics):
            total_words = refined_topics[i]['total_refined_words_en'] + refined_topics[i]['total_refined_words_cn']
            if total_words > 0:
                confidence = refined_topics[i]['refinement_rounds_completed'] / refined_topics[i]['num_refinements']
                topic_weights[i] = torch.tensor(confidence).cuda()
            else:
                topic_weights[i] = 0.0
        else:
            topic_weights[i] = 0.0
    
    return torch.sum(ot_dists * topic_weights)


def compute_embedding_cost_matrix(words1, words2, embedding_model):
    """
    Compute cost matrix between two sets of words using embeddings.
    
    Args:
        words1: List of words from current topic
        words2: List of refined words  
        embedding_model: Sentence transformer model
        
    Returns:
        cost_M: Cost matrix [len(words1), len(words2)]
    """
    # Get embeddings for both word sets
    embeddings1 = embedding_model.encode(words1)
    embeddings2 = embedding_model.encode(words2)
    
    # Convert to tensors
    emb1 = torch.tensor(embeddings1, dtype=torch.float64).cuda()
    emb2 = torch.tensor(embeddings2, dtype=torch.float64).cuda()
    
    # Compute cosine distance matrix
    # Normalize embeddings
    emb1_norm = emb1 / emb1.norm(dim=1, keepdim=True)
    emb2_norm = emb2 / emb2.norm(dim=1, keepdim=True)
    
    # Cosine similarity matrix
    cos_sim = torch.mm(emb1_norm, emb2_norm.t())
    
    # Convert to distance (1 - cosine similarity)
    cost_M = 1.0 - cos_sim
    
    # Ensure non-negative and proper shape
    cost_M = torch.clamp(cost_M, min=0.0)
    
    return cost_M