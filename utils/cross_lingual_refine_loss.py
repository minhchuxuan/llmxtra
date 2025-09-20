import torch
import numpy as np
import ot
from sentence_transformers import SentenceTransformer


def compute_cross_lingual_refine_loss(topic_probas_en, topic_probas_cn, 
                                      refined_topics, high_confidence_topics,
                                      vocab_en, vocab_cn, 
                                      embedding_model=None):
    """
    Compute refinement loss for cross-lingual topic modeling.
    
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
        # Use a lightweight multilingual model for cost computation
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    num_topics = topic_probas_en.shape[0]
    ot_dists = torch.zeros(size=(num_topics,)).cuda()
    
    for i in range(num_topics):
        if i < len(high_confidence_topics) and high_confidence_topics[i]['high_confidence_words']:
            # Get current topic words and probabilities (combined EN + CN)
            # For English part
            en_indices = torch.topk(topic_probas_en[i], k=15).indices
            en_words = [vocab_en[idx] for idx in en_indices.cpu().numpy()]
            en_probs = topic_probas_en[i].to(torch.float64)
            
            # For Chinese part  
            cn_indices = torch.topk(topic_probas_cn[i], k=15).indices
            cn_words = [vocab_cn[idx] for idx in cn_indices.cpu().numpy()]
            cn_probs = topic_probas_cn[i].to(torch.float64)
            
            # Combine current topic words and masses
            current_words = en_words + cn_words  # 30 words total
            current_masses = torch.cat([en_probs, cn_probs])  # 30 probabilities
            
            # Get refined words and their frequencies
            refined_words = high_confidence_topics[i]['high_confidence_words']
            refined_freqs = [high_confidence_topics[i]['word_frequencies'][word] 
                           for word in refined_words]
            
            if len(refined_words) > 0:
                # Compute cost matrix using embeddings
                cost_M = compute_embedding_cost_matrix(current_words, refined_words, embedding_model)
                
                # Normalize refined frequencies to create probability distribution
                refined_masses = torch.tensor(refined_freqs, dtype=torch.float64).cuda()
                refined_masses = refined_masses / refined_masses.sum()
                
                # Truncate current masses to match available words
                if len(current_words) > len(refined_words):
                    current_masses = current_masses[:len(refined_words)]
                current_masses = current_masses / current_masses.sum()
                
                # Compute optimal transport distance
                dist = ot.emd2(current_masses, refined_masses, cost_M)
                ot_dists[i] = dist
            else:
                ot_dists[i] = 0.0
        else:
            ot_dists[i] = 0.0
    
    # Compute topic weights based on refinement confidence
    topic_weights = torch.zeros(size=(num_topics,)).cuda()
    for i in range(num_topics):
        if i < len(refined_topics) and refined_topics[i]['total_refined_words'] > 0:
            # Use the number of successful refinement rounds as confidence
            confidence = refined_topics[i]['refinement_rounds_completed'] / refined_topics[i]['num_refinements']
            topic_weights[i] = torch.tensor(confidence).cuda()
        else:
            topic_weights[i] = 0.0
    
    # Compute final refinement loss
    refine_loss = torch.sum(ot_dists * topic_weights)
    
    return refine_loss


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