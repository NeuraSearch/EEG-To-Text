import os
import random
from collections import Counter
from math import floor

import numpy as np
import pickle
import torch
import math
import Networks
from gensim.models import Word2Vec

import generate_samples
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BartTokenizer, BertTokenizer
from tqdm import tqdm
#from fuzzy_match import match
#from fuzzy_match import algorithims

# macro
ZUCO_SENTIMENT_LABELS = json.load(open(r'/users/gxb18167/Datasets/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
SST_SENTIMENT_LABELS = json.load(open(r'/users/gxb18167/EEG-To-Text/dataset/stanfordsentiment/stanfordSentimentTreebank/ternary_dataset.json'))

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56, add_CLS_token = False):

    #TODO Here is where the create the function to get the word level EEG embedding
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        word_label = word_obj['content']
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            print(f'expect word eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)

        return normalize_1d(return_tensor), word_label

    #TODO Here is where the create the function to get the sentence level EEG embedding
    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    #TODO Note important none handling!
    if sent_obj is None:
        # print(f'  - skip bad sentence')   
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    input_sample['target_string'] = target_string

    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    #print(input_sample['target_ids'])
    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # get sentiment label
    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    if target_string in ZUCO_SENTIMENT_LABELS:
        input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1) # 0:Negative, 1:Neutral, 2:Positive
    else:
        input_sample['sentiment_label'] = torch.tensor(-100) # dummy value

    # get input embeddings
    word_embeddings = []
    word_embeddings_labels = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    for word in sent_obj['word']:
        #print("Word level value:", word)
        # add each word's EEG embedding as Tensors
        # check none, for v2 dataset
        if get_word_embedding_eeg_tensor(word, eeg_type, bands = bands) != None:
            word_level_eeg_tensor, word_label = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)

        else:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            # print()
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            # print()
            return None
            

        word_embeddings.append(word_level_eeg_tensor)
        word_embeddings_labels.append(word_label)
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)
    input_sample['input_embeddings_labels'] = word_embeddings_labels

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked
    

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    return input_sample

if torch.cuda.is_available():
    device = torch.device("cuda:0")

else:
    device = "cpu"

class ZuCo_dataset(Dataset):
    def create_word_label_embeddings(self, Word_Labels_List, word_embedding_dim=50):
        tokenized_words = []
        for i in range(len(Word_Labels_List)):
            tokenized_words.append([Word_Labels_List[i]])
        model = Word2Vec(sentences=tokenized_words, vector_size=word_embedding_dim, window=5, min_count=1, workers=4)
        word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        print("Number of word embeddings:", len(word_embeddings))
        # word, embedding = list(word_embeddings.items())[10]
        # print(f"Word: {word}, Embedding: {embedding}")

        Embedded_Word_labels = []
        for word in Word_Labels_List:
            Embedded_Word_labels.append(word_embeddings[word])

        return Embedded_Word_labels, word_embeddings

    def create_word_label_embeddings_contextual(self, Word_Labels_List, word_embedding_dim=50):
        tokenized_words = []
        EEG_word_level_labels = Word_Labels_List

        for i in range(len(Word_Labels_List)):
            tokenized_words.append([Word_Labels_List[i]])
        model = Word2Vec(sentences=tokenized_words, vector_size=word_embedding_dim, window=5, min_count=1, workers=4)
        word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        Embedded_Word_labels = []

        for words in range(0, len(EEG_word_level_labels)):
            current_word = EEG_word_level_labels[words]
            if current_word != "SOS" and words != len(EEG_word_level_labels) - 1:
                prior_word = EEG_word_level_labels[words - 1]

                current_word = EEG_word_level_labels[words]

                next_word = EEG_word_level_labels[words + 1]

                contextual_embedding = np.concatenate(
                    (word_embeddings[prior_word], word_embeddings[current_word], word_embeddings[next_word]), axis=-1)
                Embedded_Word_labels.append(contextual_embedding)
            elif words == len(EEG_word_level_labels) - 1:
                prior_word = EEG_word_level_labels[words - 1]
                next_word = "SOS"
                contextual_embedding = np.concatenate(
                    (word_embeddings[prior_word], word_embeddings[current_word], word_embeddings[next_word]), axis=-1)
                Embedded_Word_labels.append(contextual_embedding)

        return Embedded_Word_labels, word_embeddings

    def augment_list_balanced(self, original_list, augmentation_percentage, balance_factor=0.15):
        # Count the frequency of each word in the original list
        word_counts = {}
        for word in original_list:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Identify less frequent words
        unique_words = list(set(original_list))
        less_frequent_words = sorted(unique_words, key=lambda x: word_counts[x])

        # Determine the number of words to add for augmentation
        num_words_to_add = int(len(original_list) * augmentation_percentage)

        # Augment the list with less frequent words
        augmented_list = []
        for _ in range(num_words_to_add):
            # Choose a word with higher probability for less frequent words
            weights = [1 / (word_counts[word] ** balance_factor) for word in less_frequent_words]
            chosen_word = random.choices(less_frequent_words, weights=weights)[0]
            augmented_list.append(chosen_word)

        return augmented_list

    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent', is_add_CLS_token = False, augmentation_factor = 20, generator_name = "WGAN_v1_Text", generator_path = "", augmenation_type = 'random', text_embedding_type = "Word_Level"):
        self.inputs = []
        self.tokenizer = tokenizer

        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        else:
            device = "cpu"

        if phase == 'train' and text_embedding_type == "Word_Level":
            with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs.pkl",
                      'rb') as file:
                EEG_word_level_embeddings = pickle.load(file)
                EEG_word_level_labels = pickle.load(file)
        else:
            with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs_Sentence.pkl",
                      'rb') as file:
                EEG_word_level_embeddings = pickle.load(file)
                EEG_word_level_labels = pickle.load(file)


        z_size = 100
        if text_embedding_type == "Word_Level":
            word_embedding_dim = 50
        elif text_embedding_type == "Contextual":
            word_embedding_dim = 150
        elif text_embedding_type == "Sentence_Level":
            word_embedding_dim = 2850

        if phase == 'train':
            #print("Generator name:", generator_name)
            gen_model = Networks.get_generator_model(generator_name, z_size, word_embedding_dim)
            print(f'[INFO]loading generator model from /users/gxb18167/Datasets/Checkpoints/{generator_name}/{generator_path}')
            checkpoint = torch.load(
                fr"/users/gxb18167/Datasets/Checkpoints/{generator_name}/{generator_path}",
                map_location=device)

            print(gen_model)
            # Load the model's state_dict onto the CPU
            gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
            gen_model.to(device)
            # Set the model to evaluation mode
            gen_model.eval()

        if phase == 'train' and text_embedding_type == "Word_Level":
            Embedded_Word_labels, word_embeddings = self.create_word_label_embeddings(EEG_word_level_labels, 50)
        elif phase == 'train' and text_embedding_type == "Contextual":
            Embedded_Word_labels, word_embeddings = self.create_word_label_embeddings(EEG_word_level_labels, 50)


        #change to increase or decrease the number of synthetic samples
        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        #iterates through each task dataset in the list
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
            else:
                subjects = [subject]

            #Determines the total number of sentences on a per subject basis
            total_num_sentence = len(input_dataset_dict[subjects[0]])

            #train divider, on a per sentence count basis, 80% for training, 10% for dev, 10% for test
            train_divider = int(0.8*total_num_sentence)
            print(f"Train divider = {train_divider}")

            dev_divider = train_divider + int(0.1*total_num_sentence)

            print(f'train divider = {train_divider}')
            print(f'dev divider = {dev_divider}')

            if setting == 'unique_sent':
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    #iterates through each subject, takes 80% of that subjects sentence, and adds it to the input list
                    for key in subjects:
                        for i in range(train_divider):
                            #get_input_sample takes in each sentence dictionary
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            #print(len(input_sample))
                            if input_sample is not None:
                                #appends each subjects input sample to the input list
                                self.inputs.append(input_sample)


                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider,dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)

                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider,total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)

            elif setting == 'unique_subj':
                print('WARNING!!! only implemented for SR v1 dataset ')
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH','ZKW']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            print('++ adding task to dataset, now we have:', len(self.inputs))
        print('[INFO]input tensor size:', self.inputs[0]['input_embeddings'].size())

        if phase == 'train':
            if augmenation_type == 'random':
                Augmentation_size = floor(int(len(self.inputs) / 100 * augmentation_factor))
                print('[INFO] Augmenting Dataset by:', Augmentation_size)
                print('[INFO] Augmenting Dataset by random sampling')

                #sampled_elements = self.inputs.copy()

                sampled_elements = random.sample(self.inputs, Augmentation_size)

                #random.shuffle(sampled_elements)

                #number_of_augmented_samples = 0

                '''
                while number_of_augmented_samples < Augmentation_size:
                    for input in sampled_elements:
                        input_sample_synthetic = generate_samples.generate_synthetic_samples(input, gen_model,
                                                                                             word_embeddings,
                                                                                             EEG_word_level_embeddings)
                        if input_sample_synthetic is not None:
                            self.inputs.append(input_sample_synthetic)
                            number_of_augmented_samples += 1
                '''

                for input in sampled_elements:
                    input_sample_synthetic = generate_samples.generate_synthetic_samples(generator_name, input, gen_model,
                                                                                         word_embeddings,
                                                                                         EEG_word_level_embeddings, text_embedding_type)
                    if input_sample_synthetic is not None:
                        self.inputs.append(input_sample_synthetic)


            if augmenation_type == "TF-IDF-High" or augmenation_type == "TF-IDF-Medium" or augmenation_type == "TF-IDF-Low":
                Augmentation_size = floor(int(len(self.inputs) / 100 * augmentation_factor))
                print('[INFO] Augmenting Dataset by:', Augmentation_size)
                print('[INFO] Augmenting Dataset by TF-IDF')

                sampled_elements = random.sample(self.inputs, Augmentation_size)
                tf_idf, threshold_1, threshold_2 = self.calc_sentence_tf_idf()

                #sampled_elements = self.inputs.copy()

                #random.shuffle(sampled_elements)
                #number_of_augmented_samples = 0
                for input in sampled_elements:
                    input_sample_synthetic = generate_samples.generate_synthetic_samples_tf_idf(generator_name, input, gen_model,
                                                                                         word_embeddings,
                                                                                         EEG_word_level_embeddings, tf_idf, threshold_1, threshold_2, augmenation_type)
                    if input_sample_synthetic is not None:
                        self.inputs.append(input_sample_synthetic)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'], 
            input_sample['sent_level_EEG'],
            #input_sample['input_embeddings_labels']
        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 


    def calculate_tf_idf(self, sentences):
        # Tokenize each sentence into words and create a set of unique words
        all_words = set(word for sentence in sentences for word in sentence.split())

        # Count the frequency of each word in each sentence
        word_counts_per_sentence = [Counter(sentence.split()) for sentence in sentences]

        # Total number of sentences
        total_sentences = len(sentences)

        # Calculate TF for each word in each sentence
        tf_scores = {word: sum(word_counts[word] for word_counts in word_counts_per_sentence) / (
                    total_sentences * len(word_counts_per_sentence)) for word in all_words}

        # Calculate IDF for each word
        word_sentence_count = Counter(word for sentence in sentences for word in set(sentence.split()))
        idf_scores = {word: math.log(total_sentences / (word_sentence_count[word] + 1)) for word in all_words}

        # Calculate TF-IDF for each word
        tfidf_scores = {word: tf_scores[word] * idf_scores[word] for word in all_words}

        return tfidf_scores

    def calc_sentence_tf_idf(self):
        # To load the lists from the file:
        with open(
                r"/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs_Sentence.pkl", "rb") as file:
            EEG_word_level_embeddings = pickle.load(file)
            EEG_word_level_labels = pickle.load(file)

        list_of_sentences = []

        Sentence = []
        for i in range(len(EEG_word_level_labels)):
            if EEG_word_level_labels[i] == "SOS" and Sentence != []:
                Sentence = " ".join(Sentence)
                list_of_sentences.append(Sentence)
                Sentence = []
            elif EEG_word_level_labels[i] != "SOS":
                Sentence.append(EEG_word_level_labels[i])

        tf_idf = self.calculate_tf_idf(list_of_sentences)

        list_of_sentence_tf_idf = []

        for sentence in list_of_sentences:
            sum_of_sentence_tf_idf = 0
            for word in sentence.split():
                sum_of_sentence_tf_idf += tf_idf[word]

            sum_of_sentence_tf_idf = sum_of_sentence_tf_idf / len(sentence.split())
            list_of_sentence_tf_idf.append(sum_of_sentence_tf_idf)

        data = np.array(list_of_sentence_tf_idf)
        tertiles = np.percentile(data, [33.33, 66.67])

        threshold1 = tertiles[0]
        threshold2 = tertiles[1]

        return tf_idf, threshold1, threshold2

"""for train classifier on stanford sentiment treebank text-sentiment pairs"""
class SST_tenary_dataset(Dataset):
    def __init__(self, ternary_labels_dict, tokenizer, max_len = 56, balance_class = True):
        self.inputs = []
        
        pos_samples = []
        neg_samples = []
        neu_samples = []

        for key,value in ternary_labels_dict.items():
            tokenized_inputs = tokenizer(key, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
            input_ids = tokenized_inputs['input_ids'][0]
            attn_masks = tokenized_inputs['attention_mask'][0]
            label = torch.tensor(value)
            # count:
            if value == 0:
                neg_samples.append((input_ids,attn_masks,label))
            elif value == 1:
                neu_samples.append((input_ids,attn_masks,label))
            elif value == 2:
                pos_samples.append((input_ids,attn_masks,label))
        print(f'Original distribution:\n\tVery positive: {len(pos_samples)}\n\tNeutral: {len(neu_samples)}\n\tVery negative: {len(neg_samples)}')    
        if balance_class:
            print(f'balance class to {min([len(pos_samples),len(neg_samples),len(neu_samples)])} each...')
            for i in range(min([len(pos_samples),len(neg_samples),len(neu_samples)])):
                self.inputs.append(pos_samples[i])
                self.inputs.append(neg_samples[i])
                self.inputs.append(neu_samples[i])
        else:
            self.inputs = pos_samples + neg_samples + neu_samples
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return input_sample
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask,


'''sanity test'''
if __name__ == '__main__':

    check_dataset = 'stanford_sentiment'

    if check_dataset == 'ZuCo':
        whole_dataset_dicts = []
        
        dataset_path_task1 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-with-tokens_7-10.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        # dataset_path_task3 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-with-tokens_7-10.pickle' 
        # with open(dataset_path_task3, 'rb') as handle:
        #     whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2_v2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-with-tokens_7-15.pickle' 
        with open(dataset_path_task2_v2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))


        for key in whole_dataset_dicts[0]:
            print(f'task2_v2, sentence num in {key}:',len(whole_dataset_dicts[0][key]))


        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        dataset_setting = 'unique_sent'
        subject_choice = 'ALL'
        print(f'![Debug]using {subject_choice}')
        eeg_type_choice = 'GD'
        print(f'[INFO]eeg type {eeg_type_choice}') 
        bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
        print(f'[INFO]using bands {bands_choice}')
        train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

        print('trainset size:',len(train_set))
        print('devset size:',len(dev_set))
        print('testset size:',len(test_set))

    elif check_dataset == 'stanford_sentiment':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        SST_dataset = SST_tenary_dataset(SST_SENTIMENT_LABELS, tokenizer)
        print('SST dataset size:',len(SST_dataset))
        print(SST_dataset[0])
        print(SST_dataset[1])