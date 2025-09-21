import torch
import torch.nn.functional as F
import numpy as np
import ot
import warnings


def _torch_device_of(model, fallback):
    """Get model device safely"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback

def _ensure_fp32(x, device):
    """Ensure tensor is float32 on correct device"""
    return x.to(device=device, dtype=torch.float32)

def compute_cross_lingual_refine_loss(topic_probas_en, topic_probas_cn,
                                      topic_indices_en, topic_indices_cn,
                                      refined_topics, high_confidence_topics,
                                      vocab_en, vocab_cn,
                                      model=None):
    """
    Compute refinement loss for cross-lingual topic modeling using differentiable Sinkhorn OT.

    Args:
        topic_probas_en: English topic probabilities [num_topics, 15]
        topic_probas_cn: Chinese topic probabilities [num_topics, 15]
        topic_indices_en: Original vocabulary indices for English topics [num_topics, 15]
        topic_indices_cn: Original vocabulary indices for Chinese topics [num_topics, 15]
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
        return torch.zeros((), device=topic_probas_en.device, dtype=torch.float32)
    
    # Use consistent device/dtype
    dev = topic_probas_en.device
    topic_probas_en = _ensure_fp32(topic_probas_en, dev)
    topic_probas_cn = _ensure_fp32(topic_probas_cn, dev)
    
    num_topics = topic_probas_en.shape[0]
    ot_dists = torch.zeros(num_topics, device=dev, dtype=torch.float32)
    
    # Entropic regularization for differentiability
    reg = 0.01
    
    for i in range(num_topics):
        if (i < len(high_confidence_topics) and 
            (high_confidence_topics[i]['high_confidence_words_en'] or 
             high_confidence_topics[i]['high_confidence_words_cn'])):
            # Get current topic words and probabilities (keep gradients)
            en_indices = topic_indices_en[i]
            cn_indices = topic_indices_cn[i]
            en_words = [vocab_en[idx.item()] for idx in en_indices]
            cn_words = [vocab_cn[idx.item()] for idx in cn_indices]
            
            en_probs = topic_probas_en[i, :len(en_words)]  # tensor (grad OK)
            cn_probs = topic_probas_cn[i, :len(cn_words)]  # tensor (grad OK)
            
            # Build union vocabulary
            refined_words_en = high_confidence_topics[i]['high_confidence_words_en']
            refined_words_cn = high_confidence_topics[i]['high_confidence_words_cn']
            freq_dict_en = high_confidence_topics[i]['word_frequencies_en']
            freq_dict_cn = high_confidence_topics[i]['word_frequencies_cn']
            
            union_words = list({*en_words, *cn_words, *refined_words_en, *refined_words_cn})
            V = len(union_words)
            word2pos = {w: j for j, w in enumerate(union_words)}
            
            # Current distribution (tensor, gradient flows from en_probs/cn_probs)
            current = torch.zeros(V, device=dev, dtype=torch.float32)
            # scatter add English part
            pos_en = torch.tensor([word2pos[w] for w in en_words], device=dev)
            current.index_add_(0, pos_en, 0.5 * en_probs)
            # scatter add Chinese part
            pos_cn = torch.tensor([word2pos[w] for w in cn_words], device=dev)
            current.index_add_(0, pos_cn, 0.5 * cn_probs)
            current = current / (current.sum() + 1e-8)
            
            # Refined distribution (no grad, that's OK — it's a target)
            refined = torch.zeros(V, device=dev, dtype=torch.float32)
            if len(freq_dict_en) + len(freq_dict_cn) > 0:
                for w, f in freq_dict_en.items():
                    j = word2pos.get(w)
                    if j is not None:
                        refined[j] += 0.5 * float(f)
                for w, f in freq_dict_cn.items():
                    j = word2pos.get(w)
                    if j is not None:
                        refined[j] += 0.5 * float(f)
                refined = refined / (refined.sum() + 1e-8)
                
                # Cost matrix from embeddings (fp32 on dev)
                cost_M = compute_embedding_cost_matrix_from_precomputed(union_words, model)
                cost_M = _ensure_fp32(cost_M, dev)
                
                try:
                    # Use differentiable Sinkhorn (squared Wasserstein)
                    cost = ot.bregman.sinkhorn2(current.unsqueeze(0), refined.unsqueeze(0), cost_M, reg=reg)
                    ot_dists[i] = cost.squeeze()
                except Exception as e:
                    print(f"Warning: OT computation failed for topic {i}: {e}")
                    ot_dists[i] = 0.0
            else:
                ot_dists[i] = 0.0
        else:
            ot_dists[i] = 0.0
    
    # Topic weights from refinement progress (no grad needed)
    topic_weights = torch.zeros(num_topics, device=dev, dtype=torch.float32)
    for i in range(min(num_topics, len(refined_topics))):
        total_words = refined_topics[i]['total_refined_words_en'] + refined_topics[i]['total_refined_words_cn']
        if total_words > 0:
            num_ref = max(1, refined_topics[i]['num_refinements'])
            conf = float(refined_topics[i]['refinement_rounds_completed']) / num_ref
            topic_weights[i] = conf
    
    return (ot_dists * topic_weights).sum()


def compute_embedding_cost_matrix_from_precomputed(words, model):
    """
    Compute cost matrix between words using pre-computed word embeddings.

    Args:
        words: List of words (same list for both dimensions since it's symmetric)
        model: XTRA model with access to pre-computed word embeddings

    Returns:
        cost_M: Cost matrix [len(words), len(words)]
    """
    dev = _torch_device_of(model, torch.device('cpu'))
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
                emb_dim = model.word_embeddings_en.shape[1] if model.word_embeddings_en is not None else 768
                embeddings.append(torch.zeros(emb_dim, dtype=torch.float32, device=dev))
        # Try Chinese vocab
        elif word in word_to_idx_cn:
            idx = word_to_idx_cn[word]
            emb = model.get_word_embedding([idx], lang='cn')
            if emb is not None:
                embeddings.append(emb[0])  # Take first (and only) embedding
            else:
                # Fallback: use zero vector
                emb_dim = model.word_embeddings_cn.shape[1] if model.word_embeddings_cn is not None else 768
                embeddings.append(torch.zeros(emb_dim, dtype=torch.float32, device=dev))
        else:
            # Word not found in either vocabulary - use zero vector
            emb_dim = model.word_embeddings_en.shape[1] if model.word_embeddings_en is not None else 768
            embeddings.append(torch.zeros(emb_dim, dtype=torch.float32, device=dev))

    # Stack embeddings and ensure proper dtype/device
    embeddings_tensor = torch.stack(embeddings)
    embeddings_tensor = _ensure_fp32(embeddings_tensor, dev)

    # Compute cosine distance matrix using F.normalize
    emb_norm = F.normalize(embeddings_tensor, dim=1)
    cos_sim = emb_norm @ emb_norm.T
    cost_M = (1.0 - cos_sim).clamp_min(0.0)

    return cost_M.to(device=dev, dtype=torch.float32)