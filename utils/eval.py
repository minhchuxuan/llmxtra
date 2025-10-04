import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.io import loadmat
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import os
from typing import List, Union


def split_text_word(lines: Union[str, List[str]]) -> List[List[str]]:

    if isinstance(lines, str): # Check if input is a file path
        if not os.path.exists(lines):
            print(f"Warning: File not found at {lines}. Returning empty list.")
            return []
        try:
            with open(lines, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file {lines}: {e}. Returning empty list.")
            return []
    # Process list of strings (or lines read from file)
    return [line.strip().split() for line in lines if line.strip()]


def load_labels_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        print(f"Error: Label file not found at {path}. Returning empty array.")
        return np.array([])
    try:
        with open(path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f if line.strip()]
        return np.array(labels)
    except Exception as e:
        print(f"Error reading or parsing label file {path}: {e}. Returning empty array.")
        return np.array([])
def _cls(train_theta, test_theta, train_labels, test_labels, gamma='scale'):
    if len(np.unique(train_labels)) <= 1:
        print("Warning: Only one class present in training data. Classifier might not train meaningfully.")
        # Return default/failure metrics if training is not possible/meaningful
        return {'acc': 0.0, 'macro-F1': 0.0}

    try:
        clf = SVC(gamma=gamma, probability=False) # probability=True is slower if not needed
        clf.fit(train_theta, train_labels)
        preds = clf.predict(test_theta)
        return {
            'acc': accuracy_score(test_labels, preds),
            'macro-F1': f1_score(test_labels, preds, average='macro', zero_division=0) # Handle zero division
        }
    except Exception as e:
        print(f"Error during SVM classification: {e}")
        return {'acc': np.nan, 'macro-F1': np.nan}


def crosslingual_cls(train_theta_en, train_theta_cn,
                     test_theta_en, test_theta_cn,
                     train_labels_en, train_labels_cn,
                     test_labels_en, test_labels_cn):
    results = {
        'intra_en': _cls(train_theta_en, test_theta_en, train_labels_en, test_labels_en),
        'intra_cn': _cls(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn),
        'cross_en': _cls(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en), # Train CN, Test EN
        'cross_cn': _cls(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn), # Train EN, Test CN
    }
    return results


def crosslingual_cls_with_sim_probs(runner, train_loader, test_loader,
                                   train_labels_en, train_labels_cn,
                                   test_labels_en, test_labels_cn,
                                   temperature=0.2):
    """
    Classification using sim_probs (softmax of cosine similarity) as theta predict
    
    Args:
        runner: Runner object containing model and topic_embeddings
        train_loader: Training data loader
        test_loader: Test data loader  
        train_labels_en, train_labels_cn: Training labels
        test_labels_en, test_labels_cn: Test labels
        temperature: Temperature for softmax
    
    Returns:
        Dictionary with classification results using sim_probs
    """
    import torch.nn.functional as F
    
    # Check if topic embeddings exist
    if not hasattr(runner, 'topic_embeddings') or runner.topic_embeddings is None:
        print("Warning: No topic embeddings found. Skipping sim_probs classification.")
        return {
            'intra_en': {'acc': 0.0, 'macro-F1': 0.0},
            'intra_cn': {'acc': 0.0, 'macro-F1': 0.0},
            'cross_en': {'acc': 0.0, 'macro-F1': 0.0},
            'cross_cn': {'acc': 0.0, 'macro-F1': 0.0}
        }
    
    # Get topic embeddings from runner
    topic_embeddings = runner.topic_embeddings  # [num_topics, embedding_dim]
    model = runner.model
    
    # Get sim_probs for training data
    train_sim_probs_en = []
    train_sim_probs_cn = []
    
    # Get sim_probs for test data  
    test_sim_probs_en = []
    test_sim_probs_cn = []
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Training data
        for batch_data in train_loader:
            doc_emb_en = batch_data.get('doc_embedding_en')
            doc_emb_cn = batch_data.get('doc_embedding_cn')
            
            if doc_emb_en is not None:
                doc_emb_en = doc_emb_en.to(device)
                doc_emb_en_norm = F.normalize(doc_emb_en, p=2, dim=1)
                topic_emb_norm = F.normalize(topic_embeddings, p=2, dim=1)
                sim_en = torch.matmul(doc_emb_en_norm, topic_emb_norm.T)
                sim_probs_en = F.softmax(sim_en / temperature, dim=1)
                train_sim_probs_en.append(sim_probs_en.cpu().numpy())
                
            if doc_emb_cn is not None:
                doc_emb_cn = doc_emb_cn.to(device)
                doc_emb_cn_norm = F.normalize(doc_emb_cn, p=2, dim=1)
                topic_emb_norm = F.normalize(topic_embeddings, p=2, dim=1)
                sim_cn = torch.matmul(doc_emb_cn_norm, topic_emb_norm.T)
                sim_probs_cn = F.softmax(sim_cn / temperature, dim=1)
                train_sim_probs_cn.append(sim_probs_cn.cpu().numpy())
        
        # Test data
        for batch_data in test_loader:
            doc_emb_en = batch_data.get('doc_embedding_en')
            doc_emb_cn = batch_data.get('doc_embedding_cn')
            
            if doc_emb_en is not None:
                doc_emb_en = doc_emb_en.to(device)
                doc_emb_en_norm = F.normalize(doc_emb_en, p=2, dim=1)
                topic_emb_norm = F.normalize(topic_embeddings, p=2, dim=1)
                sim_en = torch.matmul(doc_emb_en_norm, topic_emb_norm.T)
                sim_probs_en = F.softmax(sim_en / temperature, dim=1)
                test_sim_probs_en.append(sim_probs_en.cpu().numpy())
                
            if doc_emb_cn is not None:
                doc_emb_cn = doc_emb_cn.to(device)
                doc_emb_cn_norm = F.normalize(doc_emb_cn, p=2, dim=1)
                topic_emb_norm = F.normalize(topic_embeddings, p=2, dim=1)
                sim_cn = torch.matmul(doc_emb_cn_norm, topic_emb_norm.T)
                sim_probs_cn = F.softmax(sim_cn / temperature, dim=1)
                test_sim_probs_cn.append(sim_probs_cn.cpu().numpy())
    
    # Concatenate all batches
    if train_sim_probs_en:
        train_sim_probs_en = np.concatenate(train_sim_probs_en, axis=0)
    if train_sim_probs_cn:
        train_sim_probs_cn = np.concatenate(train_sim_probs_cn, axis=0)
    if test_sim_probs_en:
        test_sim_probs_en = np.concatenate(test_sim_probs_en, axis=0)
    if test_sim_probs_cn:
        test_sim_probs_cn = np.concatenate(test_sim_probs_cn, axis=0)
    
    print(f"Train sim_probs shapes: EN {train_sim_probs_en.shape if train_sim_probs_en else 'None'}, CN {train_sim_probs_cn.shape if train_sim_probs_cn else 'None'}")
    print(f"Test sim_probs shapes: EN {test_sim_probs_en.shape if test_sim_probs_en else 'None'}, CN {test_sim_probs_cn.shape if test_sim_probs_cn else 'None'}")
    
    # Classification using sim_probs
    results = crosslingual_cls(
        train_sim_probs_en, train_sim_probs_cn,
        test_sim_probs_en, test_sim_probs_cn,
        train_labels_en, train_labels_cn,
        test_labels_en, test_labels_cn
    )
    
    return results


def print_results(results):
    """
    Print classification results in a formatted way.

    Args:
        results: Dictionary with classification metrics (output from crosslingual_cls)
    """
    for key, val in results.items():
        print(f"\n>>> {key.upper()}")
        # Check if metrics are valid numbers before formatting
        acc_str = f"{val.get('acc', 'N/A'):.4f}" if isinstance(val.get('acc'), (int, float)) and not np.isnan(val.get('acc')) else "N/A"
        f1_str = f"{val.get('macro-F1', 'N/A'):.4f}" if isinstance(val.get('macro-F1'), (int, float)) and not np.isnan(val.get('macro-F1')) else "N/A"
        print(f"  Accuracy   : {acc_str}")
        print(f"  Macro-F1   : {f1_str}")

