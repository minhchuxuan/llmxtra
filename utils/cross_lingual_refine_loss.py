import torch
import numpy as np
import ot


def get_word_embedding(word, vocab_en, vocab_cn, word_embeddings_en, word_embeddings_cn):
    """Get word embedding for a word from either English or Chinese vocabulary"""
    if word in vocab_en:
        idx = vocab_en.index(word)
        vec = word_embeddings_en[idx]
    elif word in vocab_cn:
        idx = vocab_cn.index(word)
        vec = word_embeddings_cn[idx]
    else:
        # This shouldn't happen after validation, but just in case
        raise ValueError(f"Word '{word}' not found in vocabularies")

    # Ensure we always return a torch.Tensor on CPU with float dtype
    if isinstance(vec, np.ndarray):
        vec = torch.from_numpy(vec)
    elif not torch.is_tensor(vec):
        vec = torch.tensor(vec)

    return vec.detach().cpu().float()

def compute_cross_lingual_refine_loss(topic_probas_en, topic_probas_cn,
                                      topic_indices_en, topic_indices_cn,
                                      refined_topics, high_confidence_topics,
                                      vocab_en, vocab_cn,
                                      word_embeddings_en, word_embeddings_cn,
                                      sinkhorn_reg: float = 0.1):
    """
    Compute OT loss between original and refined topic distributions
    
    Args:
        topic_probas_en: Original English probabilities [num_topics, 15]
        topic_probas_cn: Original Chinese probabilities [num_topics, 15] 
        topic_indices_en: Original English word indices [num_topics, 15]
        topic_indices_cn: Original Chinese word indices [num_topics, 15]
        high_confidence_topics: Refined words and counts from refinement
        vocab_en: English vocabulary list
        vocab_cn: Chinese vocabulary list
        word_embeddings_en: English word embeddings tensor
        word_embeddings_cn: Chinese word embeddings tensor
    
    Returns:
        OT distance between original and refined distributions
    """
    if word_embeddings_en is None or word_embeddings_cn is None:
        # No embeddings available - return zero loss
        return torch.zeros((), device=topic_probas_en.device)
    
    device = topic_probas_en.device
    num_topics = topic_probas_en.shape[0]
    total_ot_distance = torch.zeros((), device=device)
    
    for i in range(num_topics):
        if i >= len(high_confidence_topics):
            continue
            
        # 1. Get original words and probabilities (keep as torch tensors to enable backprop)
        orig_en_indices = topic_indices_en[i]
        orig_cn_indices = topic_indices_cn[i]
        orig_en_words = [vocab_en[idx.item()] for idx in orig_en_indices]
        orig_cn_words = [vocab_cn[idx.item()] for idx in orig_cn_indices]
        orig_en_probs = topic_probas_en[i]  # torch tensor on device
        orig_cn_probs = topic_probas_cn[i]  # torch tensor on device
        
        # 2. Get refined words and probabilities
        refined_en_words = high_confidence_topics[i]['high_confidence_words_en']
        refined_cn_words = high_confidence_topics[i]['high_confidence_words_cn']
        refined_en_probs = high_confidence_topics[i].get('word_probs_en', {})
        refined_cn_probs = high_confidence_topics[i].get('word_probs_cn', {})
        
        # Fallback to counts if probabilities not available
        if not refined_en_probs and not refined_cn_probs:
            refined_en_counts = high_confidence_topics[i].get('word_counts_en', {})
            refined_cn_counts = high_confidence_topics[i].get('word_counts_cn', {})
        else:
            refined_en_counts = refined_en_probs
            refined_cn_counts = refined_cn_probs
        
        # Skip if no refined words
        if not refined_en_words and not refined_cn_words:
            continue
            
        # 3. Create union vocabulary
        all_words = list(set(orig_en_words + orig_cn_words + refined_en_words + refined_cn_words))
        vocab_size = len(all_words)
        
        if vocab_size < 2:
            continue  # Need at least 2 words for meaningful OT
            
        # 4. Build distributions over union vocabulary (torch, on device)
        orig_dist = torch.zeros(vocab_size, device=device, dtype=torch.float32)
        refined_dist = torch.zeros(vocab_size, device=device, dtype=torch.float32)

        word_to_idx = {word: idx for idx, word in enumerate(all_words)}

        # Original distribution (combine EN + CN with equal weight)
        for word, prob in zip(orig_en_words, orig_en_probs):
            j = word_to_idx.get(word, None)
            if j is not None:
                orig_dist[j] = orig_dist[j] + 0.5 * prob
        for word, prob in zip(orig_cn_words, orig_cn_probs):
            j = word_to_idx.get(word, None)
            if j is not None:
                orig_dist[j] = orig_dist[j] + 0.5 * prob

        # Refined distribution (from probabilities or counts) - treated as constant targets
        if refined_en_counts or refined_cn_counts:
            for word, prob in refined_en_counts.items():
                j = word_to_idx.get(word, None)
                if j is not None:
                    refined_dist[j] = refined_dist[j] + 0.5 * float(prob)
            for word, prob in refined_cn_counts.items():
                j = word_to_idx.get(word, None)
                if j is not None:
                    refined_dist[j] = refined_dist[j] + 0.5 * float(prob)

        # Normalize distributions
        orig_sum = torch.clamp(orig_dist.sum(), min=1e-8)
        ref_sum = torch.clamp(refined_dist.sum(), min=1e-8)
        orig_dist = orig_dist / orig_sum
        refined_dist = refined_dist / ref_sum
        
        # 5. Build cost matrix using word embeddings
        cost_matrix = build_cost_matrix_torch(
            all_words, vocab_en, vocab_cn,
            word_embeddings_en, word_embeddings_cn,
            device=device
        )
        
        # 6. Compute OT distance
        try:
            # Use differentiable Sinkhorn distance (regularized OT)
            ot_distance = ot.sinkhorn2(orig_dist, refined_dist, cost_matrix, reg=sinkhorn_reg)
            # sinkhorn2 may return a tensor shape []; ensure tensor on device
            if not torch.is_tensor(ot_distance):
                ot_distance = torch.tensor(ot_distance, device=device, dtype=torch.float32)
            total_ot_distance = total_ot_distance + ot_distance
        except Exception:
            # If OT fails, skip this topic gracefully
            continue
    
    return total_ot_distance / max(1, num_topics)


def build_cost_matrix_torch(words, vocab_en, vocab_cn, word_embeddings_en, word_embeddings_cn, device):
    """
    Build cost matrix as a torch tensor using cosine distance of embeddings.
    Embeddings are detached to prevent backprop into them.
    """
    n_words = len(words)
    embeddings = []
    with torch.no_grad():
        for word in words:
            emb = get_word_embedding(word, vocab_en, vocab_cn, word_embeddings_en, word_embeddings_cn)
            embeddings.append(emb)
    embeddings = torch.stack(embeddings).to(device)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    cost_matrix = 1.0 - similarity_matrix
    cost_matrix.fill_diagonal_(0.0)
    # Detach to ensure no grad flows into embeddings
    return cost_matrix.detach()