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

import models

from torch.utils.data import DataLoader
from dataset import MIDIDataset 

def main():
    """
    Main function
    """

    '''
    Initialization: Loading and split data
    '''

    # Loading data
    train = np.load(TRAIN_DATA_MATRIX)          # shape: (11149, 460)
    validate = np.load(VALIDATE_DATA_MATRIX)    # shape: (1051, 460)
    test = np.load(TEST_DATA_MATRIX)            # shape: (1051, 460)

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

    # empty lists for saving loss values at the end of each epoch
    train_g_loss_output = []
    train_d_loss_output = []
    valid_g_loss_output = []
    valid_d_loss_output = []

    '''
    Training model: Train the model and output an example midi file at each epoch
    '''

    mmd_pitch_list = []
    mmd_duration_list = []
    mmd_rest_list = []
    MMD_pitch_old = np.inf
    MMD_duration_old = np.inf
    MMD_rest_old = np.inf
    MMD_overall_old = np.inf

    num_good_songs_best = 0
    best_epoch = 0

    # Create model
    model = models.DiffusionModel()

    print("global step = ", global_step, "max epoch = ", MAX_EPOCH)

    model_stats_saved = []

    for i in range(global_step, MAX_EPOCH):

        # Go through training data
        for (midi_tuples, syllable_embs) in dataloader_train:
            
            # Forward diffusion

            pass

        # Saving models each fifth epoch
        if i % 5 == 0:
            pass
            #print("Saving model")

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
