
import sys
import nltk
import numpy as np
import torch
nltk.download('punkt')
sys.path.insert(0, '..')

print(torch.__version__)
print("GPU Available:", torch.cuda.is_available())



def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z


def generate_samples(generator_name, g_model, input_z, input_t):
    # Create random noise as input to the generator
    # Generate samples using the generator model

    if generator_name == "DCGAN_v1" or generator_name == "DCGAN_v2" or generator_name == "WGAN_v1" or generator_name == "WGAN_v2":
        with torch.no_grad():
            g_output = g_model(input_z)
    else:
        with torch.no_grad():
            g_output = g_model(input_z, input_t)

    return g_output


def calc_sentence_tf_idf(sentence, tf_idf):
    sentence_level_tf_idf = 0
    for word in sentence:
        if word in tf_idf:
            sentence_level_tf_idf += tf_idf[word]
        else:
            return None
    return sentence_level_tf_idf/len(sentence)

def embedding_type_generation(text_embedding_type, input_embeddings_labels, word_embeddings, EEG_word_level_embeddings, generator_name, gen_model, device, original_sample_list):
    if text_embedding_type == "Word_Level":
        synthetic_EEG_samples = []
        for word in input_embeddings_labels:
            if word not in word_embeddings:
                return None

            word_embedding = word_embeddings[word]
            input_z = create_noise(1, 100, "uniform").to(device)

            word_embedding_tensor = torch.tensor(word_embedding, dtype=torch.float)
            word_embedding_tensor = word_embedding_tensor.unsqueeze(0)

            g_output = generate_samples(generator_name, gen_model, input_z, word_embedding_tensor)
            g_output = g_output.to('cpu')

            EEG_synthetic_denormalized = (g_output * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(
                EEG_word_level_embeddings)

            synthetic_sample = torch.tensor(EEG_synthetic_denormalized[0][0], dtype=torch.float).to(device)
            synthetic_sample = synthetic_sample.resize(840).to(device)
            synthetic_EEG_samples.append(synthetic_sample.to('cpu'))


    elif text_embedding_type == "Contextual":
        synthetic_EEG_samples = []
        input_embeddings_labels.insert(0, "SOS")
        for i in range(len(input_embeddings_labels)):
            current_word = input_embeddings_labels[i]

            if current_word not in word_embeddings:
                return None

            elif current_word != "SOS" and i != len(input_embeddings_labels) - 1:
                prior_word = input_embeddings_labels[i - 1]
                next_word = input_embeddings_labels[i + 1]

                #print("Current Word Embedding size", word_embeddings[current_word].shape)

                contextual_embedding = np.concatenate((word_embeddings[prior_word], word_embeddings[current_word], word_embeddings[next_word]), axis=-1)

            elif i == len(input_embeddings_labels) - 1:
                prior_word = input_embeddings_labels[i - 1]
                next_word = "SOS"

                #print("Current Word Embedding size" , word_embeddings[current_word].shape)

                contextual_embedding = np.concatenate(
                    (word_embeddings[prior_word], word_embeddings[current_word], word_embeddings[next_word]), axis=-1)

            if current_word != "SOS":
                input_z = create_noise(1, 100, "uniform").to(device)

                #print("Current Word: ", current_word)
                word_embedding_tensor = torch.tensor(contextual_embedding, dtype=torch.float)
                #print("Word Embedding Tensor: ", word_embedding_tensor.size())

                word_embedding_tensor = word_embedding_tensor.unsqueeze(0)
                #print("Word Embedding Tensor: ", word_embedding_tensor.size())

                g_output = generate_samples(generator_name, gen_model, input_z, word_embedding_tensor)
                g_output = g_output.to('cpu')

                EEG_synthetic_denormalized = (g_output * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(
                    EEG_word_level_embeddings)

                synthetic_sample = torch.tensor(EEG_synthetic_denormalized[0][0], dtype=torch.float).to(device)
                synthetic_sample = synthetic_sample.resize(840).to(device)
                synthetic_EEG_samples.append(synthetic_sample.to('cpu'))

    elif text_embedding_type == "Sentence_Level":
        max_sentence_length = 57
        synthetic_EEG_samples = []
        input_embeddings_labels.insert(0, "SOS")
        Sentence_embeddings = []
        for i in range(len(input_embeddings_labels)):
            word = input_embeddings_labels[i]
            if word not in word_embeddings:
                return None
            else:
                Sentence_embeddings.append(word_embeddings[word])
                #print("Word Embedding size: ", word_embeddings[word].shape)

        if len(Sentence_embeddings) > max_sentence_length:
            print("Sentence length is greater than 57")

        for i in range(max_sentence_length - len(Sentence_embeddings)):
            Sentence_embeddings.append(np.zeros(50))


        Combined_Sentence = np.concatenate(Sentence_embeddings, axis=0)

        input_z = create_noise(1, 100, "uniform").to(device)

        word_embedding_tensor = torch.tensor(Combined_Sentence, dtype=torch.float)
        word_embedding_tensor = word_embedding_tensor.unsqueeze(0)

        g_output = generate_samples(generator_name, gen_model, input_z, word_embedding_tensor)
        g_output = g_output.to('cpu')

        EEG_synthetic_denormalized = (g_output * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(
            EEG_word_level_embeddings)

        synthetic_sample = torch.tensor(EEG_synthetic_denormalized[0][0], dtype=torch.float).to(device)


        segments = torch.split(synthetic_sample, 8, dim=1)


        for i in range(len(input_embeddings_labels)):
            synthetic_sample = segments[i+1]
            synthetic_sample = synthetic_sample.resize(840).to(device)
            synthetic_EEG_samples.append(synthetic_sample.to('cpu'))


    synthetic_EEG_samples = torch.stack(synthetic_EEG_samples)
    padding_samples = original_sample_list[len(synthetic_EEG_samples):]
    padding_samples = padding_samples
    synthetic_EEG_samples = torch.cat((synthetic_EEG_samples, padding_samples), 0)

    return synthetic_EEG_samples


def generate_synthetic_samples(generator_name, input_sample, gen_model, word_embeddings, EEG_word_level_embeddings, text_embedding_type):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    word_embedding_dim = 50
    z_size = 100
    output_shape = (1, 105, 8)
    input_embeddings_labels = input_sample['input_embeddings_labels']
    original_sample_list = input_sample['input_embeddings']

    synthetic_EEG_samples = embedding_type_generation(text_embedding_type, input_embeddings_labels, word_embeddings, EEG_word_level_embeddings, generator_name, gen_model, device, original_sample_list)
    input_sample['input_embeddings'] = synthetic_EEG_samples

    return input_sample


def generate_synthetic_samples_tf_idf(generator_name, input_sample, gen_model, word_embeddings, EEG_word_level_embeddings, tf_idf, threshold_1, threshold_2, augmentation_type, text_embedding_type):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    input_embeddings_labels = input_sample['input_embeddings_labels']
    original_sample_list = input_sample['input_embeddings']

    sentence_tf_idf = calc_sentence_tf_idf(input_embeddings_labels, tf_idf)

    if augmentation_type == "TF-IDF-Low" and sentence_tf_idf > threshold_1:
        return None
    elif augmentation_type == "TF-IDF-Medium" and sentence_tf_idf < threshold_1 and sentence_tf_idf > threshold_2:
        return None
    elif augmentation_type == "TF-IDF-High" and sentence_tf_idf < threshold_2:
        return None

    synthetic_EEG_samples = embedding_type_generation(text_embedding_type, input_embeddings_labels, word_embeddings, EEG_word_level_embeddings, generator_name, gen_model, device, original_sample_list)
    input_sample['input_embeddings'] = synthetic_EEG_samples

    return input_sample

def generate_synthetic_samples_ablation(input_sample, word_embeddings, ablation_type):
    input_embeddings_labels = input_sample['input_embeddings_labels']

    original_sample_list = input_sample['input_embeddings']

    synthetic_EEG_samples = []

    if "ablation_noise" in ablation_type:
        for word in input_embeddings_labels:
            if word not in word_embeddings:
                return None

            #word_embedding = word_embeddings[word]
            #input_z = create_noise(1, 100, "uniform").to(device)

            random_noise = torch.randn(1, 840)

            synthetic_EEG_samples.append(random_noise)

    elif "ablation_duplicate" in ablation_type:
        for i in range(len(input_embeddings_labels)):
            word = input_embeddings_labels[i]
            if word not in word_embeddings:
                return None

            EEG_segment = original_sample_list[i]

            synthetic_EEG_samples.append(EEG_segment)

    synthetic_EEG_samples = torch.stack(synthetic_EEG_samples)
    padding_samples = original_sample_list[len(synthetic_EEG_samples):]
    padding_samples = padding_samples
    synthetic_EEG_samples = torch.cat((synthetic_EEG_samples, padding_samples), 0)

    input_sample['input_embeddings'] = synthetic_EEG_samples

    return input_sample


def generate_synthetic_samples_tf_idf_ablation(input_sample, word_embeddings, tf_idf, threshold_1, threshold_2, augmentation_type, ablation_type):

    input_embeddings_labels = input_sample['input_embeddings_labels']
    original_sample_list = input_sample['input_embeddings']

    sentence_tf_idf = calc_sentence_tf_idf(input_embeddings_labels, tf_idf)

    if "TF-IDF-Low" in augmentation_type and sentence_tf_idf > threshold_1:
        return None
    elif augmentation_type == "TF-IDF-Medium" and sentence_tf_idf < threshold_1 and sentence_tf_idf > threshold_2:
        return None
    elif augmentation_type == "TF-IDF-High" and sentence_tf_idf < threshold_2:
        return None

    synthetic_EEG_samples = []

    if "ablation_noise" in ablation_type:
        for word in input_embeddings_labels:
            if word not in word_embeddings:
                return None

            # word_embedding = word_embeddings[word]
            # input_z = create_noise(1, 100, "uniform").to(device)

            random_noise = torch.randn(1, 840)

            synthetic_EEG_samples.append(random_noise)

    elif "ablation_duplicate" in ablation_type:
        for i in range(len(input_embeddings_labels)):
            word = input_embeddings_labels[i]
            if word not in word_embeddings:
                return None

            EEG_segment = original_sample_list[i]
            synthetic_EEG_samples.append(EEG_segment)



    #synthetic_EEG_samples = embedding_type_generation(text_embedding_type, input_embeddings_labels, word_embeddings, EEG_word_level_embeddings, generator_name, gen_model, device, original_sample_list)
    input_sample['input_embeddings'] = synthetic_EEG_samples

    return input_sample






