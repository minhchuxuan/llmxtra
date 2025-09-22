import torch
import torch.nn.functional as F
import numpy as np
import ot
import warnings

# Note: POT automatically detects torch tensors and uses torch backend
# No explicit backend setting needed - POT handles this internally

# Constants for minimum support validation
MIN_SUPPORT_PER_TOPIC = 2  # Minimum total words needed for meaningful OT


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
            
            # Assert all union words are in vocabulary (strict contract enforcement)
            assert all(w in vocab_en or w in vocab_cn for w in union_words), \
                f"OOV words found in union for topic {i}: {[w for w in union_words if w not in vocab_en and w not in vocab_cn]}"
            
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
                
                # Enforce minimum support for meaningful transport
                current_support = (current > 0).sum().item()
                refined_support = (refined > 0).sum().item()
                
                if current_support < MIN_SUPPORT_PER_TOPIC or refined_support < MIN_SUPPORT_PER_TOPIC:
                    raise RuntimeError(
                        f"[OT] Topic {i}: insufficient support (current={current_support}, refined={refined_support}, need ≥{MIN_SUPPORT_PER_TOPIC})"
                    )
                
                # Cost matrix from embeddings (fp32 on dev)
                cost_M = compute_embedding_cost_matrix_from_precomputed(union_words, model)
                cost_M = _ensure_fp32(cost_M, dev)
                
                # === Strict invariants before OT ===
                # current, refined: 1-D, non-negative, float32, on the same device
                current = current.to(dtype=torch.float32, device=dev).clamp_min(0)
                refined = refined.to(dtype=torch.float32, device=dev).clamp_min(0)

                # Ensure both have identical support size V and cost is V×V on the same device
                if refined.numel() != V:
                    raise RuntimeError(f"[OT] Histogram length mismatch: current={V}, refined={refined.numel()}.")

                M = cost_M.to(dtype=torch.float32, device=dev)
                if M.dim() != 2 or M.shape[0] != V or M.shape[1] != V:
                    raise RuntimeError(f"[OT] Cost shape invalid: got {tuple(M.shape)}, expected ({V},{V}).")

                # Clean/normalize distributions (strictly 1-D)
                csum = current.sum()
                rsum = refined.sum()
                if csum <= 0 or not torch.isfinite(csum):
                    raise RuntimeError("[OT] Current histogram has zero/invalid mass.")
                if rsum <= 0 or not torch.isfinite(rsum):
                    raise RuntimeError("[OT] Refined histogram has zero/invalid mass.")

                a = (current / csum).contiguous()     # shape (V,)
                b = (refined / rsum).contiguous()     # shape (V,)

                # Cost matrix numeric hygiene
                if not torch.isfinite(M).all():
                    raise RuntimeError("[OT] Cost matrix contains non-finite values.")
                # Keep distance non-negative and diagonal well-defined
                M = M.clamp_min(0.0)
                with torch.no_grad():
                    M.fill_diagonal_(0.0)

                # Entropic regularization (fixed, explicit)
                if reg <= 0:
                    raise RuntimeError("[OT] reg must be > 0.")

                # === Call Sinkhorn with 1-D histograms ===
                # NOTE: DO NOT unsqueeze here; we want pure 1-D histograms, (V,) each.
                cost_val = ot.bregman.sinkhorn2(a, b, M, reg=reg)
                if cost_val.dim() != 0:   # ensure scalar
                    cost_val = cost_val.squeeze()

                # store cost_val downstream (no detach; keep graph)
                ot_dists[i] = cost_val.to(dtype=torch.float32, device=dev)
            else:
                raise RuntimeError(f"[OT] Topic {i}: empty frequency dictionaries (EN={len(freq_dict_en)}, CN={len(freq_dict_cn)})")
        else:
            raise RuntimeError(f"[OT] Topic {i}: empty high-confidence word lists")
    
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
    Requires all words to have valid embeddings - no zero-vector fallbacks.

    Args:
        words: List of words (same list for both dimensions since it's symmetric)
        model: XTRA model with access to pre-computed word embeddings

    Returns:
        cost_M: Cost matrix [len(words), len(words)]
    """
    dev = _torch_device_of(model, torch.device('cpu'))
    
    # Strict validation of embedding availability
    if (model.word_embeddings_en is None or model.word_embeddings_cn is None):
        raise RuntimeError("Cross-lingual OT requires precomputed EN/CN embeddings.")
    if model.word_embeddings_en.shape[1] != model.word_embeddings_cn.shape[1]:
        raise RuntimeError("EN/CN embedding dimensions must match.")

    vocab_en = model.vocab_en
    vocab_cn = model.vocab_cn
    w2i_en = {w: i for i, w in enumerate(vocab_en)}
    w2i_cn = {w: i for i, w in enumerate(vocab_cn)}

    embeddings = []
    for w in words:
        if w in w2i_en:
            emb = model.get_word_embedding([w2i_en[w]], lang='en')[0]
        elif w in w2i_cn:
            emb = model.get_word_embedding([w2i_cn[w]], lang='cn')[0]
        else:
            raise RuntimeError(f"Word '{w}' not found in EN or CN vocabulary.")
        
        if emb is None or not torch.isfinite(emb).all():
            raise RuntimeError(f"Invalid embedding for word '{w}'.")
        embeddings.append(emb.to(dev, dtype=torch.float32))

    E = torch.stack(embeddings, dim=0)                      # [V, D]
    E = torch.nn.functional.normalize(E, dim=1)             # unit vectors
    M = 1.0 - (E @ E.T).clamp(min=-1.0, max=1.0)           # cosine distance
    M.fill_diagonal_(0.0)
    if not torch.isfinite(M).all():
        raise RuntimeError("Non-finite values in cost matrix.")
    return M