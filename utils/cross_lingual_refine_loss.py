import torch
import numpy as np
import ot
import warnings


def compute_cross_lingual_refine_loss(topic_probas_en, topic_probas_cn,
                                      refined_topics, high_confidence_topics,
                                      vocab_en, vocab_cn,
                                      model=None):
    """
    Compute refinement loss for cross-lingual topic modeling using proper OT.

    Args:
        topic_probas_en: English topic probabilities [num_topics, 15]
        topic_probas_cn: Chinese topic probabilities [num_topics, 15]
        refined_topics: Output from cross-lingual refinement
        high_confidence_topics: High confidence words from refinement
        vocab_en: English vocabulary
        vocab_cn: Chinese vocabulary
        model: XTRA model with access to pre-computed word embeddings

    Returns:
        refine_loss: Scalar tensor representing refinement loss
    """
    if model is None:
        raise ValueError("XTRA model is required for word embeddings")

    # Check if word embeddings are available
    if model.word_embeddings_en is None or model.word_embeddings_cn is None:
        warnings.warn("Word embeddings not available in model. OT loss computation will use zero vectors for missing words.")
        return torch.tensor(0.0, device=topic_probas_en.device)
    
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

            # Normalize current_word_prob to sum to 1
            total_prob = sum(current_word_prob.values())
            if total_prob > 0:
                current_word_prob = {word: prob/total_prob for word, prob in current_word_prob.items()}
            
            # Get refined words and combine normalized frequency distributions
            refined_words_en = high_confidence_topics[i]['high_confidence_words_en']
            refined_words_cn = high_confidence_topics[i]['high_confidence_words_cn']
            freq_dict_en = high_confidence_topics[i]['word_frequencies_en']
            freq_dict_cn = high_confidence_topics[i]['word_frequencies_cn']

            refined_word_prob = {}
            # freq_dict_en and freq_dict_cn are already normalized (sum to 1.0 each)
            for word, freq in freq_dict_en.items():
                refined_word_prob[word] = refined_word_prob.get(word, 0) + freq
            for word, freq in freq_dict_cn.items():
                refined_word_prob[word] = refined_word_prob.get(word, 0) + freq
            
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
                
                # Compute cost matrix using pre-computed word embeddings
                cost_M = compute_embedding_cost_matrix_from_precomputed(all_words, model)
                try:
                    dist = ot.emd2(current_aligned, refined_aligned, cost_M)
                    ot_dists[i] = dist
                except Exception as e:
                    print(f"Warning: OT computation failed for topic {i}: {e}")
                    ot_dists[i] = 0.0
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


def compute_embedding_cost_matrix_from_precomputed(words, model):
    """
    Compute cost matrix between words using pre-computed word embeddings.

    Args:
        words: List of words (same list for both dimensions since it's symmetric)
        model: XTRA model with access to pre-computed word embeddings

    Returns:
        cost_M: Cost matrix [len(words), len(words)]
    """
    # Get embeddings for all words
    # First, create a mapping from word to index in the vocabulary
    vocab_en = model.vocab_en
    vocab_cn = model.vocab_cn

    # Create word to index mapping
    word_to_idx_en = {word: idx for idx, word in enumerate(vocab_en)}
    word_to_idx_cn = {word: idx for idx, word in enumerate(vocab_cn)}

    # Get embeddings for all words
    embeddings = []

    for word in words:
        # Try to find word in English vocab first
        if word in word_to_idx_en:
            idx = word_to_idx_en[word]
            emb = model.get_word_embedding([idx], lang='en')
            if emb is not None:
                embeddings.append(emb[0])  # Take first (and only) embedding
            else:
                # Fallback: use zero vector
                embeddings.append(torch.zeros(model.word_embeddings_en.shape[1], dtype=torch.float64, device=model.word_embeddings_en.device))
        # Try Chinese vocab
        elif word in word_to_idx_cn:
            idx = word_to_idx_cn[word]
            emb = model.get_word_embedding([idx], lang='cn')
            if emb is not None:
                embeddings.append(emb[0])  # Take first (and only) embedding
            else:
                # Fallback: use zero vector
                embeddings.append(torch.zeros(model.word_embeddings_cn.shape[1], dtype=torch.float64, device=model.word_embeddings_cn.device))
        else:
            # Word not found in either vocabulary - use zero vector
            emb_dim = model.word_embeddings_en.shape[1] if model.word_embeddings_en is not None else 768
            embeddings.append(torch.zeros(emb_dim, dtype=torch.float64, device=next(model.parameters()).device))

    # Stack embeddings
    embeddings_tensor = torch.stack(embeddings)

    # Compute cosine distance matrix
    # Normalize embeddings
    emb_norm = embeddings_tensor / (embeddings_tensor.norm(dim=1, keepdim=True) + 1e-8)

    # Cosine similarity matrix
    cos_sim = torch.mm(emb_norm, emb_norm.t())

    # Convert to distance (1 - cosine similarity)
    cost_M = 1.0 - cos_sim

    # Ensure non-negative and proper shape
    cost_M = torch.clamp(cost_M, min=0.0)

    return cost_M