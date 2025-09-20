import os
import yaml
import scipy.io
from scipy.io import loadmat
from runners.Runner import Runner
import argparse

from utils.data import file_utils
from utils.data.TextData import DatasetHandler
from utils import miscellaneous, seed
# from CNPMI.CNPMI import calcwcngram_complete, calcwcngram, calc_assoc
from CNPMI.CNPMI import *
from utils.TU import *
from utils.eval import *
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--weight_cluster', type=float)
    parser.add_argument('--weight_beta', type=float, default=1.0, help='Weight for InfoNCE loss')
    parser.add_argument('--weight_InfoNCE', type=float, default=1.0, help='Weight for InfoNCE theta')

    parser.add_argument('--ref_corpus_config', type=str, default="CNPMI/configs/ref_corpus/en_zh.yaml")
    parser.add_argument('--metric', type=str, default='npmi')

    parser.add_argument('--device', type=int, default=0, help='CUDA device index to use')

    # llm
    parser.add_argument('--warmStep', default=0, type=int)
    parser.add_argument('--llm_itl', action='store_true')
    parser.add_argument('--llm_step', type=int, default=3)  # the number of epochs for llm refine
    parser.add_argument('--refine_weight', type=float, default=200)
    parser.add_argument('--instruction', type=str, default='refine_labelTokenProbs',
                        choices=['refine_labelTokenProbs', 'refine_wordIntrusion'])
    parser.add_argument('--inference_bs', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=300)
    
    # Cross-lingual refinement with Gemini API
    parser.add_argument('--gemini_api_key', type=str, default=None,
                        help='Google Gemini API key for cross-lingual topic refinement')
    parser.add_argument('--refinement_rounds', type=int, default=3,
                        help='Number of self-consistent refinement rounds (R)')
    parser.add_argument('--min_frequency', type=float, default=0.01,
                        help='Minimum frequency threshold for high-confidence words')
    parser.add_argument('--refine_weight', type=float, default=0.0,
                        help='Weight for refinement loss (0 disables refinement loss)')



    # Add missing arguments used in the code
    parser.add_argument('--wandb_prj', type=str, default='Camera-ready', help='Wandb project name')

    args = parser.parse_args()
    
    # Try to get API key from environment variable if not provided via command line
    if args.gemini_api_key is None:
        args.gemini_api_key = os.getenv('GEMINI_API_KEY')

    return args


def export_beta(beta, vocab, output_prefix, lang):
    num_top_word = 15
    topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    file_utils.save_text(topic_str_list, path=f'{output_prefix}/T{num_top_word}_{lang}.txt')
    return topic_str_list

RESULT_DIR= 'output'
def main():
    args = parse_args()

    args = file_utils.update_args(args, f'./configs/model/{args.model}.yaml')
    args.warmStep = args.epochs - args.llm_step
    args = file_utils.update_args(args, f'./configs/dataset/{args.dataset}.yaml')

    if args.lang2 == "ja":
        args.ref_corpus_config = "CNPMI/configs/ref_corpus/en_ja.yaml"
        print(f"Setting reference corpus config to {args.ref_corpus_config} for Japanese")
    
    prj = args.wandb_prj 
    current_time = miscellaneous.get_current_datetime()
    output_prefix = os.path.join(RESULT_DIR + "/" + str(args.model) + "/" +str(args.dataset), 
                    str(args.weight_cluster)+"_"+str(args.weight_beta), current_time)
    miscellaneous.create_folder_if_not_exist(output_prefix)
    seed.seedEverything(args.seed)
    
    wandb.login(key="9724cd11fee33834a9fa53fea4091fab562eb4f6") #change key
    wandb.init(project=prj, config=args)
    wandb.log({'time_stamp': current_time})

    print('\n' + yaml.dump(vars(args), default_flow_style=False))
    

    dataset_handler = DatasetHandler(args.dataset, args.batch_size, args.lang1, args.lang2, args.num_topic, device=args.device)

    args.doc_embeddings_en=dataset_handler.doc_embeddings_en
    args.doc_embeddings_cn=dataset_handler.doc_embeddings_cn

    args.cluster_en=dataset_handler.clusterinfo_en
    args.cluster_cn=dataset_handler.clusterinfo_cn
    args.vocab_size_en = len(dataset_handler.vocab_en)
    args.vocab_size_cn = len(dataset_handler.vocab_cn)


    args.beta_en=dataset_handler.beta_en
    args.beta_cn=dataset_handler.beta_cn 

    args.mu_prior=dataset_handler.mu_prior
    args.var_prior = dataset_handler.var_prior



    args.vocab_en = dataset_handler.vocab_en
    args.vocab_cn = dataset_handler.vocab_cn
    

    runner = Runner(args)

    beta_en, beta_cn = runner.train(dataset_handler.train_loader)

    topic_str_list_en = export_beta(beta_en, dataset_handler.vocab_en, output_prefix, lang=args.lang1)
    topic_str_list_cn = export_beta(beta_cn, dataset_handler.vocab_cn, output_prefix, lang=args.lang2)

    for i in range(len(topic_str_list_en)):
        print(topic_str_list_en[i])
        print(topic_str_list_cn[i])

    train_theta_en, train_theta_cn = runner.test(dataset_handler.train_loader.dataset)
    test_theta_en, test_theta_cn = runner.test(dataset_handler.test_loader.dataset)

    rst_dict = {
        'beta_en': beta_en,
        'beta_cn': beta_cn,
        'train_theta_en': train_theta_en,
        'train_theta_cn': train_theta_cn,
        'test_theta_en': test_theta_en,
        'test_theta_cn': test_theta_cn,
    }

    scipy.io.savemat(f'{output_prefix}/rst.mat', rst_dict)
    
    # Calculate CNPMI
    parallel_corpus_tuples = file_utils.read_yaml(args.ref_corpus_config)['parallel_corpus_tuples']
    num_top_word = 15

    sep_token = '|'

    topics1 = read_texts_cnpmi(f'{output_prefix}/T{num_top_word}_{args.lang1}.txt')
    topics1 = split_text_word_cnpmi(topics1)
    topics2 = read_texts_cnpmi(f'{output_prefix}/T{num_top_word}_{args.lang2}.txt')
    topics2 = split_text_word_cnpmi(topics2)

    num_topic = len(topics1)
    num_top_word = len(topics1[0])

    vocab1 = set([])
    vocab2 = set([])
    word_pair_list = list()
    for k in range(num_topic):
        for i in range(num_top_word):
            w1 = topics1[k][i]
            vocab1.add(w1)
            for j in range(num_top_word):
                w2 = topics2[k][j]
                vocab2.add(w2)
                word_pair_list.append(f'{w1}{sep_token}{w2}')

    word_pair_list = tuple(word_pair_list)
    vocab1 = sorted(list(vocab1))
    vocab2 = sorted(list(vocab2))

    pool = Pool()
    for i, cp in enumerate(parallel_corpus_tuples):
        if not os.path.exists(cp[0]):
            raise FileNotFoundError(cp[0])
        if not os.path.exists(cp[1]):
            raise FileNotFoundError(cp[1])

        param_list = (cp, vocab1, vocab2, word_pair_list, sep_token)
        pool.apply_async(calcwcngram, param_list, callback=calcwcngram_complete)

    # wait for the subprocesses.
    pool.close()
    pool.join()

    topic_assoc = list()
    window_total = float(global_word_count[WTOTALKEY])
    for word_pair in word_pair_list:
        topic_assoc.append(calc_assoc(word_pair, window_total, sep_token, metric=args.metric))

    result = float(sum(topic_assoc)) / len(topic_assoc)
    print(f"CNPMI: {result:.5f}")
    wandb.log({"CNPMI": result})
    
    #Calculate TU
    texts = list()
    with open(f'{output_prefix}/T{num_top_word}_{args.lang1}.txt', 'r') as file:
        for line in file:
            texts.append(line.strip())

    tu_lang1 = TU_eva(texts)
    print(f"TU_{args.lang1}: {tu_lang1:.5f}")
    wandb.log({f"TU_{args.lang1}": tu_lang1})
    
    texts = list()
    with open(f'{output_prefix}/T{num_top_word}_{args.lang2}.txt', 'r') as file:
        for line in file:
            texts.append(line.strip())

    tu_lang2 = TU_eva(texts)
    print(f"TU_{args.lang2}: {tu_lang2:.5f}")
    wandb.log({f"TU_{args.lang2}": tu_lang2})

    tu_average = (tu_lang1 + tu_lang2) / 2
    print(f"TU_Average: {tu_average:.5f}")
    wandb.log({"TU_Average": tu_average})
    #----------Eval theta and more--------------
    dataset_name = args.dataset # Example: Change to your dataset
    model_name = args.model      # Example: Change to your model name
    num_topics = args.num_topic             # Example: Number of topics used in the model
    # num_top_words_display = 15   # Example: Number of top words in the output files (T15)

    # Construct paths
    # base_output_dir = f"./output/{dataset_name}"
    base_data_dir = f"./data/{dataset_name}"
    # mat_path = f"{base_output_dir}/{model_name}_K{num_topics}_rst.mat"
    mat_path = f'{output_prefix}/rst.mat'

    # Paths for text data and labels
    en_top_words_path = f'{output_prefix}/T{num_top_word}_{args.lang1}.txt'
    cn_top_words_path = f'{output_prefix}/T{num_top_word}_{args.lang2}.txt'

    en_corpus_path = f"{base_data_dir}/train_texts_en.txt" 
    train_labels_en_path = f"{base_data_dir}/train_labels_en.txt"
    train_labels_cn_path = f"{base_data_dir}/train_labels_cn.txt"
    test_labels_en_path = f"{base_data_dir}/test_labels_en.txt"
    test_labels_cn_path = f"{base_data_dir}/test_labels_cn.txt"
    if args.lang2 == "ja":
        train_labels_cn_path = f"{base_data_dir}/train_labels_ja.txt"
        test_labels_cn_path = f"{base_data_dir}/test_labels_ja.txt"

    print(f"--- Evaluating Model: {model_name}, Dataset: {dataset_name}, K={num_topics} ---")

    print("\n--- Loading Data ---")
    train_labels_en = load_labels_txt(train_labels_en_path)
    train_labels_cn = load_labels_txt(train_labels_cn_path)
    test_labels_en = load_labels_txt(test_labels_en_path)
    test_labels_cn = load_labels_txt(test_labels_cn_path)
    if any(arr.size == 0 for arr in [train_labels_en, train_labels_cn, test_labels_en, test_labels_cn]):
        print("Error: Failed to load one or more label files. Exiting.")
        exit()
    print("Labels loaded successfully.")

    # Load results matrix (.mat file)
    try:
        mat = loadmat(mat_path)
        train_theta_en = mat["train_theta_en"]
        train_theta_cn = mat["train_theta_cn"]
        test_theta_en = mat["test_theta_en"]    
        test_theta_cn = mat["test_theta_cn"]
        print(f"Results matrix loaded successfully from {mat_path}.")
    except FileNotFoundError:
        print(f"Error: Results matrix file not found at {mat_path}. Exiting.")
        exit()
    except KeyError as e:
        print(f"Error: Key {e} not found in results matrix {mat_path}. Exiting.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the .mat file: {e}. Exiting.")
        exit()

    print("Loading text data for Coherence/Diversity...")
    en_top_words_list = split_text_word(en_top_words_path)
    cn_top_words_list = split_text_word(cn_top_words_path)



    print("\n================= Classification =================")
    cls_results = crosslingual_cls(
        train_theta_en, train_theta_cn,
        test_theta_en, test_theta_cn,
        train_labels_en, train_labels_cn,
        test_labels_en, test_labels_cn
    )
    print_results(cls_results)
    wandb.log({"intra_en": cls_results["intra_en"]})
    wandb.log({"intra_cn": cls_results["intra_cn"]})
    wandb.log({"cross_en": cls_results["cross_en"]})
    wandb.log({"cross_cn": cls_results["cross_cn"]})




if __name__ == '__main__':
    main()
