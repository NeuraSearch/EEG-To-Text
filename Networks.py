import torch.nn as nn
import torch

import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")

else:
    device = "cpu"


class GeneratorDCGAN_v1(nn.Module):
    """
    This class is the generator for the GAN
    """

    def __init__(self, noise_dim):
        """
        The constructor for the generator

        :param noise_dim: The latent input size
        """
        super(GeneratorDCGAN_v1, self).__init__()

        self.noise_dim = noise_dim
        self.fc_noise = nn.Linear(noise_dim, 105 * 8)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, noise):
        """
        This function is the forward pass for the generator
        :param noise: The noise for the generator
        :return: Returns the synthetic EEG data
        """

        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 8)

        z = self.conv1(noise)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorDCGAN_v1(nn.Module):
    """
    The class for the discriminator
    """

    def __init__(self, n_filters):
        """
        The constructor for the discriminator
        :param n_filters:
        """
        super(DiscriminatorDCGAN_v1, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        The forward pass for the discriminator
        :param input:
        :return: returns the probability of the input being real or fake
        """
        output = self.network(input)
        return output



class GeneratorDCGAN_v2(nn.Module):
    """
    This class is the generator for the GAN
    """
    def __init__(self, noise_dim):
        """
        The constructor for the generator

        :param noise_dim: The latent input size
        """
        super(GeneratorDCGAN_v2, self).__init__()

        self.noise_dim = noise_dim

        self.fc_noise = nn.Linear(noise_dim, 105*8)  # Increase the size for more complexity
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        """
        This function is the forward pass for the generator
        :param noise: The noise for the generator
        :return: Returns the synthetic EEG data
        """
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105,8)  # Adjust the size to match conv1

        # Upsample and generate the output
        z = self.conv1(noise)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.tanh(z)

        return z

class DiscriminatorDCGAN_v2(nn.Module):
    """
    The class for the discriminator
    """
    def __init__(self, n_filters):
        super(DiscriminatorDCGAN_v2, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        The forward pass for the discriminator
        :param input:
        :return: returns the probability of the input being real or fake
        """
        output = self.network(input)
        return output




class GeneratorWGAN_v1(nn.Module):
    """
    This class is the generator for the WGAN
    """
    def __init__(self, noise_dim):
        """
        WGAN Generator constructor

        :param noise_dim: Size of the latent input
        """

        super(GeneratorWGAN_v1, self).__init__()

        self.noise_dim = noise_dim

        # Define the layers of your generator
        self.fc_noise = nn.Linear(noise_dim, 105 * 8)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)



    def forward(self, noise):
        """
        The forward pass for the generator

        :param noise: Takes in the noise for the generator
        :return: Returns the synthetic EEG data
        """

        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 8)

        # Upsample and generate the output
        z = self.conv1(noise)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorWGAN_v1(nn.Module):
    """
    This class is the discriminator for the WGAN

    """

    def __init__(self, n_filters):
        """
        The constructor for the discriminator

        :param n_filters: Takes in the number of filters
        """
        super(DiscriminatorWGAN_v1, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        The forward pass for the discriminator
        :param input: Takes in the either real or fake EEG data
        :return: Returns the probability of the input being real or fake
        """


        output = self.network(input)
        return output




class GeneratorWGAN_v2(nn.Module):
    """
    This class is the generator for the WGAN
    """
    def __init__(self, noise_dim):
        """
        WGAN Generator constructor

        :param noise_dim: Size of the latent input
        """
        super(GeneratorWGAN_v2, self).__init__()

        self.noise_dim = noise_dim
        #self.word_embedding_dim = word_embedding_dim

        self.fc_noise = nn.Linear(noise_dim, 105*8)  # Increase the size for more complexity

        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        """
        The forward pass for the generator

        :param noise: Takes in the noise for the generator
        :return: Returns the synthetic EEG data
        """
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105,8)  # Adjust the size to match conv1

        # Upsample and generate the output
        z = self.conv1(noise)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.tanh(z)

        return z

class DiscriminatorWGAN_v2(nn.Module):
    """
    This class is the discriminator for the WGAN

    """
    def __init__(self, n_filters):
        """
        The constructor for the discriminator

        :param n_filters: Takes in the number of filters
        """

        super(DiscriminatorWGAN_v2, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        The forward pass for the discriminator
        :param input: Takes in the either real or fake EEG data
        :return: Returns the probability of the input being real or fake
        """
        output = self.network(input)
        return output

class GeneratorDCGAN_v1_Text(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(GeneratorDCGAN_v1_Text, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

        self.fc_noise = nn.Linear(noise_dim, 105 * 8)
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 8)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, noise, word_embedding):
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 8)

        # Process word embedding
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)

        # Concatenate noise and word embedding
        combined_input = torch.cat([noise, word_embedding], dim=1)

        # Upsample and generate the output
        z = self.conv1(combined_input)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorDCGAN_v1_Text(nn.Module):
    def __init__(self, n_filters, word_embedding_dim):
        super(DiscriminatorDCGAN_v1_Text, self).__init__()

        self.word_embedding_dim = word_embedding_dim
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 8)

        self.network = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input, word_embedding):
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)

        combined_input = torch.cat([input, word_embedding], dim=1)

        output = self.network(combined_input)
        return output


class GeneratorDCGAN_v2_Text(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(GeneratorDCGAN_v2_Text, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

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
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
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

class DiscriminatorDCGAN_v2_Text(nn.Module):
    def __init__(self, n_filters, word_embedding_dim):
        super(DiscriminatorDCGAN_v2_Text, self).__init__()

        self.word_embedding_dim = word_embedding_dim
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 8)

        self.network = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input, word_embedding):
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)

        combined_input = torch.cat([input, word_embedding], dim=1)

        output = self.network(combined_input)
        return output



class GeneratorWGAN_v1_Text(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(GeneratorWGAN_v1_Text, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

        self.fc_noise = nn.Linear(noise_dim, 105 * 8)
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 8)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)



    def forward(self, noise, word_embedding):
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 8)

        # Process word embedding
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)

        # Concatenate noise and word embedding
        combined_input = torch.cat([noise, word_embedding], dim=1)

        #print("Gen Combined Input:", combined_input.shape)
        # Upsample and generate the output
        z = self.conv1(combined_input)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorWGAN_v1_Text(nn.Module):
    def __init__(self, n_filters, word_embedding_dim):
        super(DiscriminatorWGAN_v1_Text, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.word_embedding_dim = word_embedding_dim
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 8)

        self.network = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input, word_embedding):
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)

        combined_input = torch.cat([input, word_embedding], dim=1)
        #print("combined_input:", combined_input.shape)

        output = self.network(combined_input)
        return output


class GeneratorWGAN_v2_Text(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(GeneratorWGAN_v2_Text, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

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
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
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

class DiscriminatorWGAN_v2_Text(nn.Module):
    def __init__(self, n_filters, word_embedding_dim):
        super(DiscriminatorWGAN_v2_Text, self).__init__()

        self.word_embedding_dim = word_embedding_dim
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 8)

        self.network = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input, word_embedding):
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)

        combined_input = torch.cat([input, word_embedding], dim=1)

        output = self.network(combined_input)
        return output



class GeneratorDCGAN_v1_Sentence(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(GeneratorDCGAN_v1_Sentence, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

        self.fc_noise = nn.Linear(noise_dim, 105 * 456)
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 456)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, noise, word_embedding):
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 456)

        # Process word embedding
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 456)

        print("word_embedding:", word_embedding.shape)
        print("input:", noise.shape)

        # Concatenate noise and word embedding
        combined_input = torch.cat([noise, word_embedding], dim=1)

        print("combined_input:", combined_input.shape)

        # Upsample and generate the output
        z = self.conv1(combined_input)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorDCGAN_v1_Sentence(nn.Module):
    def __init__(self, n_filters, word_embedding_dim):
        super(DiscriminatorDCGAN_v1_Sentence, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.word_embedding_dim = word_embedding_dim
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 456)

        self.network = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (456 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input, word_embedding):
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 456)


        combined_input = torch.cat([input, word_embedding], dim=1)

        output = self.network(combined_input)
        return output


class GeneratorDCGAN_v2_Sentence(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(GeneratorDCGAN_v2_Sentence, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

        self.fc_noise = nn.Linear(noise_dim, 105*456)  # Increase the size for more complexity
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105*456)  # Increase the size for more complexity
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
        noise = noise.view(noise.size(0), 1, 105,456)  # Adjust the size to match conv1

        # Process word embedding
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 456)  # Adjust the size to match conv1

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

class DiscriminatorDCGAN_v2_Sentence(nn.Module):
    def __init__(self, n_filters, word_embedding_dim):
        super(DiscriminatorDCGAN_v2_Sentence, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.word_embedding_dim = word_embedding_dim
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 456)

        self.network = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (456 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input, word_embedding):
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 456)


        combined_input = torch.cat([input, word_embedding], dim=1)

        output = self.network(combined_input)
        return output


class GeneratorWGAN_v1_Sentence(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(GeneratorWGAN_v1_Sentence, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

        self.fc_noise = nn.Linear(noise_dim, 105 * 456)
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 456)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)



    def forward(self, noise, word_embedding):
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 456)

        # Process word embedding
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 456)

        # Concatenate noise and word embedding
        combined_input = torch.cat([noise, word_embedding], dim=1)


        # Upsample and generate the output
        z = self.conv1(combined_input)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorWGAN_v1_Sentence(nn.Module):
    def __init__(self, n_filters, word_embedding_dim):
        super(DiscriminatorWGAN_v1_Sentence, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.word_embedding_dim = word_embedding_dim
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 456)

        self.network = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (456 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input, word_embedding):
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 456)


        combined_input = torch.cat([input, word_embedding], dim=1)

        output = self.network(combined_input)
        return output


class GeneratorWGAN_v2_Sentence(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim):
        super(GeneratorWGAN_v2_Sentence, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

        self.fc_noise = nn.Linear(noise_dim, 105*456)  # Increase the size for more complexity
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105*456)  # Increase the size for more complexity
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
        noise = noise.view(noise.size(0), 1, 105,456)  # Adjust the size to match conv1

        # Process word embedding
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 456)  # Adjust the size to match conv1

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

class DiscriminatorWGAN_v2_Sentence(nn.Module):
    def __init__(self, n_filters, word_embedding_dim):
        super(DiscriminatorWGAN_v2_Sentence, self).__init__()

        self.word_embedding_dim = word_embedding_dim
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105 * 456)

        self.network = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (456 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input, word_embedding):
        word_embedding = self.fc_word_embedding(word_embedding.to(self.device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 456)

        combined_input = torch.cat([input, word_embedding], dim=1)

        output = self.network(combined_input)
        return output



class GeneratorACGAN_v1(nn.Module):
    def __init__(self, noise_dim, n_classes=5860):
        super(GeneratorACGAN_v1, self).__init__()

        self.noise_dim = noise_dim
        self.label_emb = nn.Embedding(n_classes, 100)

        self.fc_noise = nn.Linear(noise_dim, 105 * 8)
        self.fc_word_embedding = nn.Linear(100, 105 * 8)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, noise, word_embedding):
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 8)

        # Process word embedding
        word_embedding = self.label_emb(word_embedding)
        word_embedding = self.fc_word_embedding(word_embedding)
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)

        # Concatenate noise and word embedding
        combined_input = torch.cat([noise, word_embedding], dim=1)

        # Upsample and generate the output
        z = self.conv1(combined_input)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorACGAN_v1(nn.Module):
    def __init__(self, n_filters):
        super(DiscriminatorACGAN_v1, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample

        )

        self.adv_layer = nn.Sequential(
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Softmax()
        )

    def forward(self, input):
        #print("combined_input:", combined_input.shape)

        output = self.network(input)
        valid = self.adv_layer(output)
        label = self.aux_layer(output)


        return valid, label


class GeneratorACGAN_v2(nn.Module):
    def __init__(self, noise_dim, n_classes=5860):
        super(GeneratorACGAN_v2, self).__init__()

        self.noise_dim = noise_dim
        self.label_emb = nn.Embedding(n_classes, 100)

        # Define the layers of your generator
        self.fc_noise = nn.Linear(noise_dim, 105 * 8)
        self.fc_word_embedding = nn.Linear(100, 105 * 8)


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
        noise = noise.view(noise.size(0), 1, 105, 8)

        # Process word embedding
        word_embedding = self.label_emb(word_embedding)
        word_embedding = self.fc_word_embedding(word_embedding)
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)

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

def get_generator_model(Model_name, z_size, word_embedding_dim):
    print("Model_name:", Model_name)

    if Model_name == 'WGAN_v1_Text' or "BERT":
        return GeneratorWGAN_v1_Text(z_size, word_embedding_dim)
    if Model_name == 'DCGAN_v1':
        return GeneratorDCGAN_v1(z_size)
