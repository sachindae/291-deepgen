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
from modules import UNet, UNet_wText, UNet_wText_OH

from torch.utils.data import DataLoader
from dataset import MIDIDataset, MIDIDataset_OH

from sklearn.preprocessing import OneHotEncoder

# Run using the following:
# python lstm-gan-lyrics2melody-v2-torch.py --settings_file settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEST_MIDI_FILE = "valid_melodies\melody{epoch}-{sample_num}.mid"

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

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
    """
    dataset_train = MIDIDataset(train, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES)
    dataset_valid = MIDIDataset(validate, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES)
    dataset_test = MIDIDataset(test, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)
    """
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train[:, :NUM_MIDI_FEATURES*SONGLENGTH])
    
    dataset_train_OH = MIDIDataset_OH(train, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES, enc)
    dataset_valid_OH = MIDIDataset_OH(validate, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES, enc)
    dataset_test_OH = MIDIDataset_OH(test, NUM_MIDI_FEATURES, SONGLENGTH, NUM_SYLLABLE_FEATURES, enc)
    dataloader_train = DataLoader(dataset_train_OH, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid_OH, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset_valid_OH, batch_size=BATCH_SIZE, shuffle=True)
    
    # Epoch counter initialization
    global_step = 0

    '''
    Training model: Train the model and output an example midi file at each epoch
    '''

    num_good_songs_best = 0
    best_epoch = 0

    ### Hyperparams
    lr = 3e-4

    #################################
    #  Create model
    #################################
    #diffusion = Diffusion(device=device)
    diffusion = Diffusion(device=device, melody_len=636)
    #model = UNet(device=device).to(device)
    #model = UNet_wText(device=device).to(device)
    model = UNet_wText_OH(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    print("global step = ", global_step, "max epoch = ", MAX_EPOCH)
    model_stats_saved = []
    pbar = tqdm(range(global_step, MAX_EPOCH))
    for epoch in pbar:

        # Go through training data
        model.train()
        train_loss = 0
        for (midi_tuples, syllable_embs) in dataloader_train:
            midi_tuples = midi_tuples.to(device)
            syllable_embs = syllable_embs.to(device)

            t = diffusion.sample_timesteps(midi_tuples.shape[0]).to(device)
            x_t, noise = diffusion.noise_melodies(midi_tuples, t)
            predicted_noise = model(x_t, t, syllable_embs)
            #predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        pbar.set_postfix(MSE=train_loss / len(dataloader_train))

        # Sample some melodies and output them
        if epoch > 391:
            sample_num = 0
            for (midi_tuples, syllable_embs) in dataloader_valid:
                syllable_embs = syllable_embs.to(device)
                sampled_melodies = diffusion.sample_wText(model, syllable_embs, n=syllable_embs.shape[0]).cpu().detach()
                for i,sampled_melody in enumerate(sampled_melodies):
                    sampled_melody = sampled_melody.transpose(1, 0)[0].transpose(1, 0).reshape((1,636*3))
                    sampled_melody = dataset_valid_OH.inverse(sampled_melody).reshape(20,3)
                    midi_melody = dataset_valid_OH.create_midi_pattern_from_discretized_data(sampled_melody)
                    destination = DEST_MIDI_FILE.format(epoch=epoch, sample_num=sample_num)
                    midi_melody.write(destination)
                    print(f'Melody {sample_num}:, {sampled_melody}')
                    
                    """
                    # Rearrange tensor to be of shape [melody length, 3]
                    sampled_melody = sampled_melody.transpose(1, 0)[0].transpose(1, 0).numpy()
                    denormed_melody = dataset_valid.denormalize(sampled_melody)
                    discretized_melody = dataset_train.discretize(denormed_melody)
                    midi_melody = dataset_train.create_midi_pattern_from_discretized_data(discretized_melody)
                    destination = DEST_MIDI_FILE.format(epoch=epoch, sample_num=sample_num)
                    midi_melody.write(destination)
                    print(f'Melody {sample_num}:, {denormed_melody}')
                    #print(f'Discrete Melody {i}:, {discretized_melody}')
                    """
                    sample_num += 1
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
