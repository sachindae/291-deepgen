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
from dilated import DiffWave, DiffWave_wText
from modules import UNet

from torch.utils.data import DataLoader
from dataset import MIDIDataset 

import matplotlib.pyplot as plt

import midi_statistics

# Run using the following:
# python lstm-gan-lyrics2melody-v2-torch.py --settings_file settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAINEDMODEL = "trained_models\model_epoch{epoch}.pt"
TRAININGMELODY = "training_melodies\melody{epoch}"

lyrics_list = [[['Then','Then'],['the','the'],['rain','rainstorm'],['storm','rainstorm'],['came','came'],
          ['ov','over'],['er','over'],['me','me'],['and','and'],['i','i'],['felt','felt'],['my','my'],
          ['spi','spirit'],['rit','spirit'],['break','break']],
          [['E','Everywhere'],['very','Everywhere'],['where','Everywhere'],['I','I'],['look','look'],
         ['I','I'],['found','found'],['you','you'],['look','looking'],['king','looking'],['back','back']],
            [['Must','Must'],['have','have'],['been','been'],['love','love'],
          ['but','but'],['its','its'],['o','over'],['ver','over'],['now','now'],['lay','lay'],['a','a'],
          ['whis','whisper'],['per','whisper'],['on','on'],['my','my'],['pil','pillow'],['low','pillow']],
            [['You','You'],['turn','turn'],['my','my'],['nights','nights'],
          ['in','into'],['to','into'],['days','days'],['Lead','Lead'],['me','me'],['mys','mysterious'],['te','mysterious'],
          ['ri','mysterious'],['ous','mysterious'],['ways','ways']]]

syll_model_path = '.\enc_models\syllEncoding_20190419.bin'
word_model_path = '.\enc_models\wordLevelEncoder_20190419.bin'
syllModel = Word2Vec.load(syll_model_path)
wordModel = Word2Vec.load(word_model_path)

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def lyrics_encode(lyr_list):
    enc_lyr = []
    
    for lyrics in lyr_list:
        length_song = len(lyrics)
        cond = []
        
        for i in range(20):
            if i < length_song:
                syll2Vec = syllModel.wv[lyrics[i][0]]
                word2Vec = wordModel.wv[lyrics[i][1]]
                cond.append(np.concatenate((syll2Vec,word2Vec)))
            else:
                cond.append(np.concatenate((syll2Vec,word2Vec)))
        
        
        flattened_cond = []
        for x in cond:
            for y in x:
                flattened_cond.append(y)
        enc_lyr.append(flattened_cond)
    return np.array(enc_lyr)

def lyrics_to_sentence(lyr):
    # input: [['Then','Then'],['the','the'],['rain','rainstorm'],['storm','rainstorm'],['came','came'],
    #          ['ov','over'],['er','over'],['me','me'],['and','and'],['i','i'],['felt','felt'],['my','my'],
    #          ['spi','spirit'],['rit','spirit'],['break','break']],
    # output: Then the rainstorm came over me and i felt my spirit break
    sentence = ""
    for w, s in lyr:
        sentence += w + " "
    return sentence

# Function to save model
def save_model(epoch_num, mod, opt):

    # Save model
    root_model_path = TRAINEDMODEL.format(epoch=str(epoch_num))
    print(root_model_path)
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
    #model = DiffWave().to(device) #UNet().to(device)
    model = DiffWave_wText(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    # Load previous model if flag used
    load = False
    if load:
        model_name = TRAINEDMODEL.format(epoch=1550)
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
            #predicted_noise = model(x_t, t)
            predicted_noise = model(x_t, t, syllable_embs.to(device))
            
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        pbar.set_postfix(MSE=train_loss / len(dataloader_train))

        # Sample some melodies and output them
        if epoch % 50 == 0:
            #sampled_melodies = diffusion.sample(model, n=5).cpu().detach()
            #sampled_melodies = diffusion.sample_1d(model, n=5).cpu().detach()
            sample_lyrics = torch.Tensor(lyrics_encode(lyrics_list)).to(device)
            #sampled_melodies = diffusion.sample_1d_wText(model, syllable_embs, n=syllable_embs.shape[0]).cpu().detach()
            sampled_melodies = diffusion.sample_1d_wText(model, sample_lyrics, n=sample_lyrics.shape[0]).cpu().detach()
            for i,sampled_melody in enumerate(sampled_melodies):
                sampled_melody = sampled_melody.transpose(1, 0).numpy() # -> [melody len, 3]
                denormed_melody = dataset_train.denormalize2(sampled_melody)


                #denormed_melody = np.concatenate((denormed_melody, np.ones(denormed_melody.shape), np.zeros(denormed_melody.shape)), axis=1)
                denormed_melody = dataset_train.discretize(denormed_melody)
                midi_melody = dataset_train.create_midi_pattern_from_discretized_data(denormed_melody)
                destination = TRAININGMELODY.format(epoch=epoch)+str(i)+".mid"
                print(f'Melody {i}, Lyrics = {lyrics_to_sentence(lyrics_list[i])} \n {denormed_melody}')
                midi_melody.write(destination)
  
                tuned_melody = midi_statistics.tune_song(denormed_melody)
                midi_melody_tuned = dataset_train.create_midi_pattern_from_discretized_data(tuned_melody)
                destination = TRAININGMELODY.format(epoch=epoch)+str(i)+"_tuned.mid"
                midi_melody_tuned.write(destination)

            save_model(epoch, model, optimizer)

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
