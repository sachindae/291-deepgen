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
from torchtext.data.metrics import bleu_score
import pickle

# Run using the following:
# python lstm-gan-lyrics2melody-v2-torch.py --settings_file settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAINEDMODEL = "trained_models\model_epoch{epoch}.pt"
TRAININGMELODY = "training_melodies\melody{epoch}"
FILE_SAVE_EPOCH = ".\saved_models\epoch_models\model_epoch"
FILE_SAVE_SAVE = ".\saved_models\saved_model"
FILE_SAVE_PITCH = ".\saved_models\saved_model_best_pitch_mmd"
FILE_SAVE_DURATION=".\saved_models\saved_model_best_duration_mmd"
FILE_SAVE_REST = ".\saved_models\saved_model_best_rest_mmd"
FILE_SAVE_OVERALL = ".\saved_models\saved_model_best_overall_mmd"
FILE_SAVE_BLEU = ".\saved_models\saved_model_best_bleu_sum"
FILE_SAVE_END = ".\saved_models\saved_model_end_of_training"
DATA_SAVE_METRIC = ".\data\metric.pkl"
DATA_SAVE_STATS = ".\data\stats.pkl"
SONGLENGTH = 20
NUM_MIDI_FEATURES = 3

"""
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

"""
lyrics_list = [
    [['I','I'],['wan','wanna'],['na','wanna'],['feel','feel'],['like','like'],['strand','stranded'],['ed','stranded'],['You','You'],['left','left'],['me','me'],
     ['a','alone'],['lone','alone'],['I','I'],['can','can'],['take','take'],['it','it'],['no','no'],['more','more'],['I','I'],['wish','wish']],
    [['here','here'],['by','by'],['my','my'],['side','side'],['I','I'],['re','remember'],['mem','remember'],['ber','remember'],['those','those'],['days','days'],
     ['when','when'],['it','it'],['was','was'], ['a','a'], ['time','time'] , ['if','if'], ['I','I'], ['could','could'], ['right','right'], ['the','the']],
    [['Good','Goodbye'],['bye','Goodbye'],['Nor','Norma'],['ma','Norma'],['Jean','Jean'],['Though','Though'],['I','I'],['knew','knew'],['you','you'],['at','at'],
     ['al','all'],['l','all'],['You','You'],['had','had'],['the','the'],['grace','grace'],['hold','hold'],['your','yourself'],['self','yourself'],['While','While']],
    [['out','out'],['in','in'],['to','to'],['your','your'],['bra','brain'],['in','brain'],['They','They'],['set','set'],['you','you'],['on','on'],
     ['the','the'],['tread','treadmill'],['mill','treadmill'],['And','And'],['they','they'],['made','made'],['you','you'],['change','change'],['your','your'],['And','And']],
    [['So', 'So'],['I', 'I'],['m', 'm'],['gon', 'gonna'],['na', 'gonna'],['love', 'love'],['you', 'you'],['like', 'like'],['I', 'I'],['m', 'm'],
     ['gon', 'gonna'],['na', 'gonna'], ['lose', 'lose'],['you', 'you'],['I', 'I'],['m', 'm'],['gon', 'gonna'],['na', 'gonna'],['hold', 'hold'],['you', 'you']]
    ]
 

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
def save_model(mod, opt, root_model_path):

    # Save model
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

    best = {}
    best['num_good_songs_best'] = 0
    best['best_epoch'] = 0
    best['MMD_pitch_old'] = np.inf
    best['MMD_duration_old'] = np.inf
    best['MMD_rest_old'] = np.inf
    best['MMD_overall_old'] = np.inf
    best['BLEU'] = 0
    
    metrics = {}
    metrics['num_good_songs'] = []
    metrics['MMD_pitch'] = []
    metrics['MMD_duration'] = []
    metrics['MMD_rest'] = []
    metrics['MMD_overall'] = []
    metrics['BLEU_pitch'] = []
    metrics['BLEU_duration'] = []
    metrics['BLEU_rest'] = []
    metrics['BLEU_sum'] = []
    metrics['mse'] = []


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

            save_model(model, optimizer, TRAINEDMODEL.format(epoch=str(epoch)))
            
        
        # do it every epoch - takes long!!!    
        model_stats, validation_songs = get_model_stats(model, diffusion, dataloader_valid, dataset_valid)
        model_stats_saved.append(model_stats)
        best, metrics = save_best_models(best, metrics, epoch, model_stats, validation_songs, validate, dataset_valid, model, optimizer)
        metrics['mse'].append(train_loss)

        with open(DATA_SAVE_METRIC, 'wb') as file:
            pickle.dump(metrics, file)
        with open(DATA_SAVE_STATS, 'wb') as file:
            pickle.dump(model_stats_saved, file)
            
    save_model(model, optimizer, FILE_SAVE_END)
    
    return 0

def get_model_stats(model, diffusion, dataloader, dataset):
        
    model_stats = {}
    model_stats['stats_scale_tot'] = 0
    model_stats['stats_repetitions_2_tot'] = 0
    model_stats['stats_repetitions_3_tot'] = 0
    model_stats['stats_span_tot'] = 0
    model_stats['stats_unique_tones_tot'] = 0
    model_stats['stats_avg_rest_tot'] = 0
    model_stats['num_of_null_rest_tot'] = 0
    model_stats['best_scale_score'] = 0
    model_stats['best_repetitions_2'] = 0
    model_stats['best_repetitions_3'] = 0
    model_stats['num_perfect_scale'] = 0
    model_stats['num_good_songs'] = 0
    
    validation_songs = []
    for (midi_tuples, syllable_embs) in dataloader:
        #midi_tuples = midi_tuples.to(device)
        #midi_tuples = midi_tuples.squeeze(2)

        sampled_melodies = diffusion.sample_1d_wText(model, syllable_embs, n=syllable_embs.shape[0]).cpu().detach()
        
        for i, sampled_melody in enumerate(sampled_melodies):
            sampled_melody = sampled_melody.transpose(1, 0).numpy() # -> [melody len, 3]
            denormed_melody = dataset.denormalize2(sampled_melody)

            #denormed_melody = np.concatenate((denormed_melody, np.ones(denormed_melody.shape), np.zeros(denormed_melody.shape)), axis=1)
            denormed_melody = dataset.discretize(denormed_melody)
            tuned_melody = midi_statistics.tune_song(denormed_melody)

            validation_songs.append(tuned_melody)

            stats = midi_statistics.get_all_stats(tuned_melody)
            model_stats['stats_scale_tot'] += stats['scale_score']
            model_stats['stats_repetitions_2_tot'] += float(stats['repetitions_2'])
            model_stats['stats_repetitions_3_tot'] += float(stats['repetitions_3'])
            model_stats['stats_unique_tones_tot'] += float(stats['tones_unique'])
            model_stats['stats_span_tot'] += stats['tone_span']
            model_stats['stats_avg_rest_tot'] += stats['average_rest']
            model_stats['num_of_null_rest_tot'] += stats['num_null_rest']
            model_stats['best_scale_score'] = max(stats['scale_score'], model_stats['best_scale_score'])
            model_stats['best_repetitions_2'] = max(stats['repetitions_2'], model_stats['best_repetitions_2'])
            model_stats['best_repetitions_3'] = max(stats['repetitions_3'], model_stats['best_repetitions_3'])

            # if stats['scale_score'] == 1.0:
            #    model_stats['num_perfect_scale'] += 1

            if stats['scale_score'] == 1.0 and stats['tones_unique'] > 3 \
               and stats['tone_span'] > 4 and stats['num_null_rest'] > 8 and stats['tone_span'] < 13\
               and stats['repetitions_2'] > 4:
                model_stats['num_good_songs'] += 1
    print(validation_songs[0])
    print(validation_songs[1])
    
    print(model_stats)
    return model_stats, validation_songs


def save_song(dataset, song, filename):
    midi_melody = dataset.create_midi_pattern_from_discretized_data(song)
    midi_melody.write(filename)
    return 0
    
def save_best_models(best, metrics, epoch, model_stats, validation_songs, validate, dataset, model, optimizer):

    metrics['num_good_songs'].append(model_stats['num_good_songs'])        
    if model_stats['num_good_songs'] > best['num_good_songs_best']:
        save_model(model, optimizer, FILE_SAVE_SAVE)
        print('NEW MODEL SAVED!\n')
        best['best_epoch'] = epoch
        # np.save('saved_model/train_data.npy', train)
        # np.save('saved_model/valid_data.npy', validate)
        # np.save('saved_model/test_data.npy', test)
        best['num_good_songs_best'] = model_stats['num_good_songs']

    print('Best ratio of good songs, ', best['num_good_songs_best'], ' at epoch', best['best_epoch'])

    print("MMD2=========================================================================================\n")
    val_gen_pitches = np.zeros((np.shape(validation_songs)[0],SONGLENGTH))
    val_dat_pitches = np.zeros((np.shape(validate)[0],SONGLENGTH))
    val_gen_duration = np.zeros((np.shape(validation_songs)[0],SONGLENGTH))
    val_dat_duration = np.zeros((np.shape(validate)[0],SONGLENGTH))
    val_gen_rests = np.zeros((np.shape(validation_songs)[0],SONGLENGTH))
    val_dat_rests = np.zeros((np.shape(validate)[0],SONGLENGTH))

    print(np.shape(validation_songs), np.shape(val_gen_pitches))

    for i in range(SONGLENGTH):
        val_gen_pitches[:, i] = np.array(validation_songs)[:, i, 0]
        val_gen_duration[:, i] = np.array(validation_songs)[:, i, 1]
        val_gen_rests[:, i] = np.array(validation_songs)[:, i, 2]
        val_dat_pitches[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i]
        val_dat_duration[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i + 1]
        val_dat_rests[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i + 2]

    MMD_pitch = mmd.Compute_MMD(val_gen_pitches, val_dat_pitches)
    metrics['MMD_pitch'].append(MMD_pitch)
    print("MMD pitch:", MMD_pitch)
    if MMD_pitch < best['MMD_pitch_old']:
        print("New lowest value of MMD for pitch", MMD_pitch)
        save_model(model, optimizer, FILE_SAVE_PITCH)
        best['MMD_pitch_old'] = MMD_pitch
        save_song(dataset, validation_songs[0], FILE_SAVE_PITCH+str(epoch)+"_"+str(0)+".mid")
        save_song(dataset, validation_songs[1], FILE_SAVE_PITCH+str(epoch)+"_"+str(1)+".mid")

    MMD_duration = mmd.Compute_MMD(val_gen_duration,val_dat_duration)
    metrics['MMD_duration'].append(MMD_duration)
    print("MMD duration:", MMD_duration)
    if MMD_duration < best['MMD_duration_old']:
        print("New lowest value of MMD for duration", MMD_duration)
        save_model(model, optimizer, FILE_SAVE_DURATION)
        best['MMD_duration_old'] = MMD_duration
        save_song(dataset, validation_songs[0], FILE_SAVE_DURATION+str(epoch)+"_"+str(0)+".mid")
        save_song(dataset, validation_songs[1], FILE_SAVE_DURATION+str(epoch)+"_"+str(1)+".mid")

    MMD_rest = mmd.Compute_MMD(val_gen_rests,val_dat_rests)
    metrics['MMD_rest'].append(MMD_rest)
    print("MMD rest:", MMD_rest)
    if MMD_rest < best['MMD_rest_old']:
        print("New lowest value of MMD for rest", MMD_rest)
        save_model(model, optimizer, FILE_SAVE_REST)
        best['MMD_rest_old'] = MMD_rest
        save_song(dataset, validation_songs[0], FILE_SAVE_REST+str(epoch)+"_"+str(0)+".mid")
        save_song(dataset, validation_songs[1], FILE_SAVE_REST+str(epoch)+"_"+str(1)+".mid")

    MMD_overall = MMD_rest + MMD_duration + MMD_pitch
    metrics['MMD_overall'].append(MMD_overall)
    print("MMD overall:", MMD_overall)
    if MMD_overall < best['MMD_overall_old']:
        print("New lowest value of MMD for overall", MMD_overall)
        save_model(model, optimizer, FILE_SAVE_OVERALL)
        best['MMD_overall_old'] = MMD_overall
        save_song(dataset, validation_songs[0], FILE_SAVE_OVERALL+str(epoch)+"_"+str(0)+".mid")
        save_song(dataset, validation_songs[1], FILE_SAVE_OVERALL+str(epoch)+"_"+str(1)+".mid")
    
    #get bleu
    val_dat_pitches = np.expand_dims(val_dat_pitches,axis=1)
    val_dat_duration = np.expand_dims(val_dat_duration,axis=1)
    val_dat_rests = np.expand_dims(val_dat_rests,axis=1)
    bleu_pitches = bleu_score(val_gen_pitches.astype(str), val_dat_pitches.astype(str))
    bleu_duration = bleu_score(val_gen_duration.astype(str), val_dat_duration.astype(str))
    bleu_rests = bleu_score(val_gen_rests.astype(str), val_dat_rests.astype(str))
    bleu_sum = bleu_pitches + bleu_duration + bleu_rests
    metrics['BLEU_pitch'].append(bleu_pitches)
    metrics['BLEU_duration'].append(bleu_duration)
    metrics['BLEU_rest'].append(bleu_rests)
    metrics['BLEU_sum'].append(bleu_sum)
    print(f"BLEU: pitch - {bleu_pitches}, duration - {bleu_duration}, rests - {bleu_rests}, sum - {bleu_sum}")  
    if bleu_sum > best['BLEU']:
        print("New BLEU sum", bleu_sum)
        save_model(model, optimizer, FILE_SAVE_BLEU)
        best['BLEU'] = bleu_sum
        save_song(dataset, validation_songs[0], FILE_SAVE_BLEU+str(epoch)+"_"+str(0)+".mid")
        save_song(dataset, validation_songs[1], FILE_SAVE_BLEU+str(epoch)+"_"+str(1)+".mid")        
    
    return best, metrics
    

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
