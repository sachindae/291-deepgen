import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.ops import array_ops
import torch
from shutil import copyfile
import utils
import midi_statistics
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import datetime
import shutil
import mmd
from tqdm import tqdm
import random

from models import Diffusion #, UNet_1D
from dilated import DiffWave
from modules import UNet

from torch.utils.data import DataLoader
from dataset import MIDIDataset 

import matplotlib.pyplot as plt

# Run using the following:
# python lstm-gan-lyrics2melody-v2-torch.py --settings_file settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Function to save model
def save_model(epoch_num, mod, opt):

    # Save model
    root_model_path = 'trained_models/model_epoch' + str(epoch_num) + '.pt'
    model_dict = mod.state_dict()
    state_dict = {'model': model_dict, 'optimizer': opt.state_dict()}
    torch.save(state_dict, root_model_path)

    print('Saved model')

def main():
    """
    Main function
    """

    '''
    Initialization: Loading and split data
    '''

    # Loading data
    train = np.load(TRAIN_DATA_MATRIX)
    validate = np.load(VALIDATE_DATA_MATRIX)
    test = np.load(TEST_DATA_MATRIX)

    print("Training set: ", np.shape(train)[0], " songs, Validation set: ", np.shape(validate)[0], " songs, "
          "Test set: ", np.shape(test)[0], " songs.")

    # Load datasets
    dataset_train = MIDIDataset(train, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES)
    dataset_valid = MIDIDataset(validate, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES)
    dataset_test = MIDIDataset(test, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)

    # Epoch counter initialization
    global_step = 0

    '''
    Training model: Train the model and output an example midi file at each epoch
    '''

    num_good_songs_best = 0
    best_epoch = 0

    ### Hyperparams
    input_size = 1
    lr = 9e-4

    #################################
    #  Create model
    #################################
    diffusion = Diffusion()
    model = DiffWave().to(device) #UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    # Load previous model if flag used
    load = False
    if load:
        model_name = 'trained_models/model_epoch4000.pt'
        state_dict = torch.load(model_name)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        print('Model loaded!', model_name)

    print("global step = ", global_step, "max epoch = ", MAX_EPOCH)
    model_stats_saved = []
    pbar = tqdm(range(global_step, MAX_EPOCH))
    for epoch in pbar:

        # Go through training data
        model.train()
        train_loss = 0
        for (midi_tuples, syllable_embs) in dataloader_train:
            midi_tuples = midi_tuples.to(device)
            midi_tuples = midi_tuples.squeeze(2)

            t = diffusion.sample_timesteps(midi_tuples.shape[0]).to(device)
            #x_t, noise = diffusion.noise_melodies(midi_tuples, t)
            x_t, noise = diffusion.noise_melodies_1d(midi_tuples, t)
            predicted_noise = model(x_t, t)
            
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        pbar.set_postfix(MSE=train_loss / len(dataloader_train))

        # Sample some melodies and output them
        if epoch % 50 == 0:
            #sampled_melodies = diffusion.sample(model, n=5).cpu().detach()
            sampled_melodies = diffusion.sample_1d(model, n=5).cpu().detach()
            for i,sampled_melody in enumerate(sampled_melodies):
                sampled_melody = sampled_melody.transpose(1, 0).numpy() # -> [melody len, 3]
                denormed_melody = dataset_train.denormalize(sampled_melody)
                midi_melody = dataset_train.create_midi_pattern_from_discretized_data(denormed_melody)
                destination = f"training_melodies/melody{epoch}.mid"
                print(f'Melody {i}:, {denormed_melody}')
                midi_melody.write(destination)
                break

    # Save model

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a GAN to generate sequential, real-valued data.')
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str,
                        default='settings')
    settings = vars(parser.parse_args())
    if settings['settings_file']:
        settings = utils.load_settings_from_file(settings)
    for (k, v) in settings.items():
        print(v, '\t', k)
    locals().update(settings)
    main()
