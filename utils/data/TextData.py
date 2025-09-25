import os
import numpy as np
import scipy
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from . import file_utils


class _BilingualDataset(Dataset):
    def __init__(self, bow_en, bow_cn, doc_embeddings_en=None, doc_embeddings_cn=None):
        self.bow_en = bow_en
        self.bow_cn = bow_cn
        self.doc_embeddings_en = doc_embeddings_en
        self.doc_embeddings_cn = doc_embeddings_cn
        self.bow_size_en = len(self.bow_en)
        self.bow_size_cn = len(self.bow_cn)

    def __len__(self):
        return max(self.bow_size_en, self.bow_size_cn)

    def __getitem__(self, index):
        en_idx = index % self.bow_size_en
        cn_idx = index % self.bow_size_cn
        
        return_dict = {
            'bow_en': self.bow_en[en_idx],
            'bow_cn': self.bow_cn[cn_idx]
        }
        
        # Add document embeddings if available
        if self.doc_embeddings_en is not None:
            return_dict['doc_embedding_en'] = self.doc_embeddings_en[en_idx]
        if self.doc_embeddings_cn is not None:
            return_dict['doc_embedding_cn'] = self.doc_embeddings_cn[cn_idx]
            
        return return_dict


class CrosslingualDataset:
    def __init__(self, dataset_dir, lang1, lang2, dict_path, device='cpu', batch_size=200, as_tensor=True):
        self.batch_size = batch_size

        self.train_texts_en, self.test_texts_en, self.train_bow_en, self.test_bow_en, self.train_labels_en, self.test_labels_en, self.vocab_en, self.word2id_en, self.id2word_en = self.read_data(dataset_dir, lang=lang1)
        self.train_texts_cn, self.test_texts_cn, self.train_bow_cn, self.test_bow_cn, self.train_labels_cn, self.test_labels_cn, self.vocab_cn, self.word2id_cn, self.id2word_cn = self.read_data(dataset_dir, lang=lang2)

        self.train_size_en = len(self.train_texts_en)
        self.train_size_cn = len(self.train_texts_cn)
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)

        self.trans_dict, self.trans_matrix_en, self.trans_matrix_cn = self.parse_dictionary(dict_path)

        # Load document embeddings
        self.doc_embeddings_en = np.load(os.path.join(dataset_dir, f'doc_embeddings_{lang1}_train.npy'))
        self.doc_embeddings_cn = np.load(os.path.join(dataset_dir, f'doc_embeddings_{lang2}_train.npy'))
        
        # Load word embeddings
        self.word_embeddings_en = np.load(os.path.join(dataset_dir, f'word_embeddings_{lang1}.npy'))
        self.word_embeddings_cn = np.load(os.path.join(dataset_dir, f'word_embeddings_{lang2}.npy'))

        self.pretrained_WE_en = scipy.sparse.load_npz(os.path.join(dataset_dir, f'word2vec_{lang1}.npz')).toarray()
        self.pretrained_WE_cn = scipy.sparse.load_npz(os.path.join(dataset_dir, f'word2vec_{lang2}.npz')).toarray()

        self.Map_en2cn = self.get_Map(self.trans_matrix_en, self.train_bow_en)
        self.Map_cn2en = self.get_Map(self.trans_matrix_cn, self.train_bow_cn)

        if as_tensor:
            self.train_bow_en = self.move_to_device(self.train_bow_en, device)
            self.test_bow_en = self.move_to_device(self.test_bow_en, device)
            self.train_bow_cn = self.move_to_device(self.train_bow_cn, device)
            self.test_bow_cn = self.move_to_device(self.test_bow_cn, device)
            
            # Move document embeddings to device
            self.doc_embeddings_en = self.move_to_device(self.doc_embeddings_en, device)
            self.doc_embeddings_cn = self.move_to_device(self.doc_embeddings_cn, device)

            self.train_dataloader = DataLoader(
                _BilingualDataset(
                    self.train_bow_en, 
                    self.train_bow_cn,
                    self.doc_embeddings_en,
                    self.doc_embeddings_cn
                ), 
                batch_size=batch_size, 
                shuffle=True
            )
            self.test_dataloader = DataLoader(
                _BilingualDataset(
                    self.test_bow_en, 
                    self.test_bow_cn
                ), 
                batch_size=batch_size, 
                shuffle=False
            )

    def move_to_device(self, bow, device):
        return torch.as_tensor(bow, device=device).float()

    def read_data(self, dataset_dir, lang):
        train_texts = file_utils.read_text(os.path.join(dataset_dir, 'train_texts_{}.txt'.format(lang)))
        test_texts = file_utils.read_text(os.path.join(dataset_dir, 'test_texts_{}.txt'.format(lang)))
        vocab = file_utils.read_text(os.path.join(dataset_dir, 'vocab_{}'.format(lang)))
        word2id = dict(zip(vocab, range(len(vocab))))
        id2word = dict(zip(range(len(vocab)), vocab))

        train_bow = scipy.sparse.load_npz(os.path.join(dataset_dir, 'train_bow_matrix_{}.npz'.format(lang))).toarray()
        test_bow = scipy.sparse.load_npz(os.path.join(dataset_dir, 'test_bow_matrix_{}.npz'.format(lang))).toarray()

        train_labels = np.loadtxt(f'{dataset_dir}/train_labels_{lang}.txt').astype('int32')
        test_labels = np.loadtxt(f'{dataset_dir}/test_labels_{lang}.txt').astype('int32')

        return train_texts, test_texts, train_bow, test_bow, train_labels, test_labels, vocab, word2id, id2word

    def parse_dictionary(self, dict_path):
        trans_dict = defaultdict(set)

        trans_matrix_en = np.zeros((self.vocab_size_en, self.vocab_size_cn), dtype='int32')
        trans_matrix_cn = np.zeros((self.vocab_size_cn, self.vocab_size_en), dtype='int32')

        dict_texts = file_utils.read_text(dict_path)

        for line in dict_texts:
            terms = (line.strip()).split()
            if len(terms) == 2:
                cn_term = terms[0]
                en_term = terms[1]
                if cn_term in self.word2id_cn and en_term in self.word2id_en:
                    trans_dict[cn_term].add(en_term)
                    trans_dict[en_term].add(cn_term)
                    cn_term_id = self.word2id_cn[cn_term]
                    en_term_id = self.word2id_en[en_term]

                    trans_matrix_en[en_term_id][cn_term_id] = 1
                    trans_matrix_cn[cn_term_id][en_term_id] = 1

        return trans_dict, trans_matrix_en, trans_matrix_cn

    def get_Map(self, trans_matrix, bow):
        Map = (trans_matrix * bow.sum(0)[:, np.newaxis]).astype('float32')
        Map = Map + 1
        Map_sum = np.sum(Map, axis=1)
        t_index = Map_sum > 0
        Map[t_index, :] = Map[t_index, :] / Map_sum[t_index, np.newaxis]

        return Map


class DatasetHandler:
    """
    Wrapper around CrosslingualDataset to maintain compatibility with existing main.py code
    """
    def __init__(self, dataset_name, batch_size, lang1, lang2, num_topics, device=0):
        # Construct dataset directory path
        dataset_dir = f'data/{dataset_name}'
        
        # Construct dictionary path based on languages
        if lang2 == "ja":
            dict_path = 'data/dict/MUSE/ja-en.txt'
        else:
            dict_path = 'data/dict/ch_en_dict.dat'
        
        # Create CrosslingualDataset instance
        self.dataset = CrosslingualDataset(
            dataset_dir=dataset_dir,
            lang1=lang1,
            lang2=lang2,
            dict_path=dict_path,
            device=f'cuda:{device}' if torch.cuda.is_available() else 'cpu',
            batch_size=batch_size,
            as_tensor=True
        )
        
        # Expose all attributes from the dataset for compatibility
        for attr_name in dir(self.dataset):
            if not attr_name.startswith('_'):
                setattr(self, attr_name, getattr(self.dataset, attr_name))
        
        # Create train and test loaders with proper naming
        self.train_loader = self.dataset.train_dataloader
        self.test_loader = self.dataset.test_dataloader