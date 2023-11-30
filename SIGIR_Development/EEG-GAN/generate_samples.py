import torch
import torch.nn as nn
import sys
import nltk
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
nltk.download('punkt')
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
sys.path.insert(0, '..')
import pickle
from torch.autograd import grad as torch_grad

print(torch.__version__)
print("GPU Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"

def create_word_label_embeddings(Word_Labels_List, word_embedding_dim=50):
    tokenized_words = []
    for i in range(len(Word_Labels_List)):
        tokenized_words.append([Word_Labels_List[i]])
    model = Word2Vec(sentences=tokenized_words, vector_size=word_embedding_dim, window=5, min_count=1, workers=4)
    word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
    print("Number of word embeddings:", len(word_embeddings))
    #word, embedding = list(word_embeddings.items())[10]
    #print(f"Word: {word}, Embedding: {embedding}")

    Embedded_Word_labels = []
    for word in Word_Labels_List:
        Embedded_Word_labels.append(word_embeddings[word])

    return Embedded_Word_labels, word_embeddings

def create_dataloader(EEG_word_level_embeddings, Embedded_Word_labels):
    #EEG_word_level_embeddings_normalize = (EEG_word_level_embeddings - np.mean(EEG_word_level_embeddings)) / np.std(EEG_word_level_embeddings)

    # Assuming EEG_synthetic is the generated synthetic EEG data
    #EEG_synthetic_denormalized = (EEG_synthetic * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(EEG_word_level_embeddings)


    EEG_word_level_embeddings_normalize = (EEG_word_level_embeddings - np.mean(EEG_word_level_embeddings)) / np.max(np.abs(EEG_word_level_embeddings))


    float_tensor = torch.tensor(EEG_word_level_embeddings_normalize, dtype=torch.float)
    float_tensor = float_tensor.unsqueeze(1)

    #print(EEG_word_level_embeddings_normalize)
    # Calculate mean and standard deviation
    print(torch.isnan(float_tensor).any())

    train_data = []
    for i in range(len(float_tensor)):
       train_data.append([float_tensor[i], Embedded_Word_labels[i]])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64)
    return trainloader

def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z

def generate_samples(g_model, input_z, input_t):
    # Create random noise as input to the generator
    # Generate samples using the generator model
    with torch.no_grad():
        g_output = g_model(input_z, input_t)

    return g_output.cpu().numpy()

class Generator(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

        # Define the layers of your generator
        self.fc_noise = nn.Linear(noise_dim, 105*8)  # Increase the size for more complexity
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105*8)  # Increase the size for more complexity
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise, word_embedding):
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105,8)  # Adjust the size to match conv1

        # Process word embedding
        word_embedding = self.fc_word_embedding(word_embedding.to(device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)  # Adjust the size to match conv1

        # Concatenate noise and word embedding
        combined_input = torch.cat([noise, word_embedding], dim=1)

        # Upsample and generate the output
        z = self.conv1(combined_input)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.tanh(z)

        return z

def generate_synthetic_samples(input_sample, gen_model, word_embeddings, EEG_word_level_embeddings):
    word_embedding_dim = 50
    z_size = 100
    output_shape = (1, 105, 8)
    input_embeddings_labels = input_sample['input_embeddings_labels']
    original_sample_list = input_sample['input_embeddings']

    input_sample = list(input_sample)

    synthetic_EEG_samples = []
    for word in input_embeddings_labels:
        word_embedding = word_embeddings[word]
        input_z = create_noise(1, 100, "uniform")

        word_embedding_tensor = torch.tensor(word_embedding, dtype=torch.float)
        word_embedding_tensor = word_embedding_tensor.unsqueeze(0)

        g_output = generate_samples(gen_model, input_z, word_embedding_tensor)
        EEG_synthetic_denormalized = (g_output * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(
            EEG_word_level_embeddings)

        synthetic_sample = torch.tensor(EEG_synthetic_denormalized[0][0], dtype=torch.float)

        synthetic_sample = synthetic_sample.resize(840)
        synthetic_EEG_samples.append(synthetic_sample)

    synthetic_EEG_samples = torch.stack(synthetic_EEG_samples)
    padding_samples = original_sample_list[len(synthetic_EEG_samples):]
    synthetic_EEG_samples = torch.cat((synthetic_EEG_samples, padding_samples), 0)

    input_sample[0] = synthetic_EEG_samples
    input_sample = tuple(input_sample)

    return input_sample






