import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.XTRA import XTRA
from utils.cross_lingual_refinement import refine_cross_lingual_topics
from utils.cross_lingual_refine_loss import compute_cross_lingual_refine_loss


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = XTRA(args)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.device}" if args.device is not None else "cuda:0")
            self.model = self.model.to(self.device)

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.learning_rate
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.args.lr_scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma)
        else:
            raise NotImplementedError(self.args.lr_scheduler)

        return lr_scheduler

    def train(self, data_loader):

        data_size = len(data_loader.dataset)
        num_batch = len(data_loader)
        optimizer = self.make_optimizer()

        # Add these parameters
        max_grad_norm = 1.0  # Adjust this value as needed
 
        if 'lr_scheduler' in self.args:
            lr_scheduler = self.make_lr_scheduler(optimizer)

        for epoch in range(1, self.args.epochs + 1):
            # Phase 2: Check if we should extract topic words
            print(self.args.warmStep)
            if epoch >= self.args.warmStep:
                # Extract topic words from current beta
                beta_en, beta_cn = self.model.get_beta()
                topic_words_en, topic_words_cn = self.get_topic_words(beta_en, beta_cn, topk=50)
                print(f"Phase 2 - Epoch {epoch}: Extracted topic words")
                print(f"English topic words: {len(topic_words_en)} topics")
                print(f"Chinese topic words: {len(topic_words_cn)} topics")
                
                # Step 1: Extract top-k words using torch.topk for each topic
                topk = 15  # Number of top words to keep
                
                # For English topics
                top_values_en, top_indices_en = torch.topk(beta_en, topk, dim=1)
                # Step 2: Rescale probabilities to sum to 1 using torch.div
                topic_probas_en = torch.div(top_values_en, top_values_en.sum(dim=1, keepdim=True))
                
                # For Chinese topics  
                top_values_cn, top_indices_cn = torch.topk(beta_cn, topk, dim=1)
                # Step 2: Rescale probabilities to sum to 1 using torch.div
                topic_probas_cn = torch.div(top_values_cn, top_values_cn.sum(dim=1, keepdim=True))
                
                print(f"Created clean probability distributions over top {topk} words")
                print(f"English topic_probas shape: {topic_probas_en.shape}")
                print(f"Chinese topic_probas shape: {topic_probas_cn.shape}")
                
                # Cross-lingual topic refinement using Gemini API
                refined_topics, high_confidence_topics = None, None
                if hasattr(self.args, 'gemini_api_key') and self.args.gemini_api_key:
                    print("Starting cross-lingual topic refinement...")
                    
                    refined_topics, high_confidence_topics = refine_cross_lingual_topics(
                        topic_words_en=topic_words_en,
                        topic_words_cn=topic_words_cn,
                        topic_probas_en=topic_probas_en,
                        topic_probas_cn=topic_probas_cn,
                        api_key=self.args.gemini_api_key,
                        R=getattr(self.args, 'refinement_rounds', 3),
                        min_frequency=getattr(self.args, 'min_frequency', 0.01)  # Lower threshold: 1% instead of 10%
                    )
                    
                    print(f"Refined {len(refined_topics)} topics using cross-lingual refinement")
                    
                    # Print summary of refined topics
                    for i, (refined, high_conf) in enumerate(zip(refined_topics, high_confidence_topics)):
                        total_words = len(high_conf['high_confidence_words_en']) + len(high_conf['high_confidence_words_cn'])
                        print(f"Topic {i}: {total_words} high-confidence words ({len(high_conf['high_confidence_words_en'])} EN, {len(high_conf['high_confidence_words_cn'])} CN)")
                        sample_words = high_conf['high_confidence_words_en'][:3] + high_conf['high_confidence_words_cn'][:3]
                        print(f"  Sample words: {', '.join(sample_words[:5])}...")

                else:
                    print("No Gemini API key provided, skipping cross-lingual refinement")
                #TO-do: add loss ot 

            sum_loss = 0.

            loss_rst_dict = defaultdict(float)
            print(epoch)


            self.model.train()
            for batch_data in data_loader:
                batch_bow_en = batch_data['bow_en']
                batch_bow_cn = batch_data['bow_cn']
                cluster_info = {
                'cluster_en': batch_data['cluster_en'],
                'cluster_cn': batch_data['cluster_cn']
                }
                document_info = {
                'doc_embedding_en': batch_data['doc_embedding_en'],
                'doc_embedding_cn': batch_data['doc_embedding_cn']
                }
            
                # Trong Runner.py, train method:
                rst_dict = self.model(batch_bow_en, batch_bow_cn, document_info, cluster_info)
                batch_loss = rst_dict['loss']
                
                # Add refinement loss if we have refined topics
                if (epoch >= self.args.warmStep and 
                    refined_topics is not None and 
                    high_confidence_topics is not None and
                    hasattr(self.args, 'refine_weight') and 
                    self.args.refine_weight > 0):
                    
                    try:
                        refine_loss = compute_cross_lingual_refine_loss(
                            topic_probas_en=topic_probas_en,
                            topic_probas_cn=topic_probas_cn,
                            refined_topics=refined_topics,
                            high_confidence_topics=high_confidence_topics,
                            vocab_en=self.model.vocab_en,
                            vocab_cn=self.model.vocab_cn,
                            model=self.model
                        )
                        
                        if refine_loss > 0:
                            weighted_refine_loss = self.args.refine_weight * refine_loss
                            batch_loss = batch_loss + weighted_refine_loss
                            rst_dict['refine_loss'] = refine_loss
                            
                    except Exception as e:
                        print(f"Warning: Failed to compute refinement loss: {e}")

                for key in rst_dict:
                    if 'loss' in key:
                        loss_rst_dict[key] += rst_dict[key]

                optimizer.zero_grad()
                batch_loss.backward()
                # Add gradient clipping before optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

                sum_loss += batch_loss.item() * len(batch_bow_en)

            if 'lr_scheduler' in self.args:
                lr_scheduler.step()

            sum_loss /= data_size

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / num_batch :.3f}'

            print(output_log)


        beta_en, beta_cn = self.model.get_beta()
        beta_en = beta_en.detach().cpu().numpy()
        beta_cn = beta_cn.detach().cpu().numpy()
        return beta_en, beta_cn


    def get_theta(self, bow, lang):
        """Get topic distribution from BOW"""
        theta_list = list()
        data_size = bow.shape[0]
        all_idx = torch.split(torch.arange(data_size,), self.args.batch_size)
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_bow = bow[idx]
                # Đảm bảo batch_bow ở đúng device
                if isinstance(batch_bow, np.ndarray):
                    batch_bow = torch.tensor(batch_bow, dtype=torch.float)
                if hasattr(batch_bow, 'device') and batch_bow.device != self.device:
                    batch_bow = batch_bow.to(self.device)
                theta = self.model.get_theta(batch_bow, lang=lang)
                theta_list.extend(theta.detach().cpu().numpy().tolist())

        return np.asarray(theta_list)

    # Keep existing code and modify Runner.py:
    def test(self, dataset):
        theta_en = self.get_theta(dataset.bow_en, lang='en')
        theta_cn = self.get_theta(dataset.bow_cn, lang='cn')
        return theta_en, theta_cn

    def get_topic_words(self, beta_en, beta_cn, topk_refine=50, topk_loss=15):
        """Extract top words for each topic from beta matrices

        Args:
            beta_en: English beta matrix
            beta_cn: Chinese beta matrix
            topk_refine: Number of words for refinement vocabulary (default: 50)
            topk_loss: Number of words for loss computation (default: 15)

        Returns:
            topic_words_en, topic_words_cn: Lists of topic word strings
        """
        # Convert CUDA tensors to numpy arrays if necessary
        if torch.is_tensor(beta_en):
            beta_en = beta_en.detach().cpu().numpy()
        if torch.is_tensor(beta_cn):
            beta_cn = beta_cn.detach().cpu().numpy()

        topic_words_en = []
        topic_words_cn = []

        # Get vocabularies from the model
        vocab_en = self.model.vocab_en
        vocab_cn = self.model.vocab_cn

        # Extract top words for each topic in English
        for i, topic_dist in enumerate(beta_en):
            top_word_indices = np.argsort(topic_dist)[:-(topk_refine + 1):-1]
            topic_words = np.array(vocab_en)[top_word_indices]
            topic_str = ' '.join(topic_words)
            topic_words_en.append(topic_str)

        # Extract top words for each topic in Chinese
        for i, topic_dist in enumerate(beta_cn):
            top_word_indices = np.argsort(topic_dist)[:-(topk_refine + 1):-1]
            topic_words = np.array(vocab_cn)[top_word_indices]
            topic_str = ' '.join(topic_words)
            topic_words_cn.append(topic_str)

        return topic_words_en, topic_words_cn