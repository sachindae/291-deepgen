import torch
import os
import random

import cv2
import numpy as np

from torch.utils.data import Dataset

import utils
import midi
import pretty_midi

import matplotlib.pyplot as plt

class MIDIDataset(Dataset):

    def __init__(self, data, MIDI_TUPLE_SIZE, SONG_LENGTH, SYLLABLE_EMB_DIM):
        '''
        data: [N, SONG_LENGTH * (MIDI_TUPLE_SIZE + SYLLABLE_EMB DIM)] matrix
              where N is number of data, SONG_LENGTH is number
              of notes in song, and SYLLABLE_EMB_DIM dim is dimensionality
              of syllable embeddings. First SONG_LENGTH * MIDI_TUPLE_SIZE elements
              correspond to the MIDI tuples, remaining elements
              correspond to syllable embeddings of paired lyrics
        '''

        self.MIDI_TUPLE_SIZE = MIDI_TUPLE_SIZE
        self.SONG_LENGTH = SONG_LENGTH
        self.SYLLABLE_EMB_DIM = SYLLABLE_EMB_DIM

        # Params for normalization
        self.MIDI_VAL_MAX = 127
        self.DUR_MAX = 32
        self.REST_MAX = 32

        # Mean center the data
        # means, stds, ch_minmaxes = [], [], []
        # for channel in range(3):
        #     channel_data = data[:,channel:60:3]
        #     means.append(channel_data.mean())
        #     stds.append(channel_data.std())
        #     data[:,channel:60:3] = (channel_data - means[-1]) / stds[-1]
        #     ch_min, ch_max = data[:,channel:60:3].min(), data[:,channel:60:3].max()
        #     ch_minmaxes.append((ch_min, ch_max))
        #     data[:,channel:60:3] = (data[:,channel:60:3] - ch_min) / (ch_max - ch_min)
        #     data[:,channel:60:3] = 2 * data[:,channel:60:3] - 1
        # self.means = means
        # self.stds = stds
        # self.ch_minmaxes = ch_minmaxes

        # Go through data splitting MIDI tuples and paired lyrics (syllable embeddings)
        self.samples = []
        k = 0
        for d in data:
            k += 1
            # Separate data vector into MIDI tuples and syllable embeddings 
            midi_tuples = np.array(np.split(d[:MIDI_TUPLE_SIZE*SONG_LENGTH], SONG_LENGTH))
            syllable_embs = d[MIDI_TUPLE_SIZE*SONG_LENGTH:]

            # Store some training MIDI files
            if k < 100:
                midi_melody = self.create_midi_pattern_from_discretized_data(midi_tuples)
                destination = f"training_melodies/test{k}.mid"
                #midi_melody.write(destination)

            # Normalize MIDI tuples (0-127, 0-32, 0-32) to [-1, 1]
            midi_tuples = self.normalize2(midi_tuples).transpose(1, 0)
#            midi_tuples = midi_tuples.transpose(1, 0)
            midi_tuples = np.expand_dims(midi_tuples, axis=1)

            # Create sample
            sample = (midi_tuples, syllable_embs)
            self.samples.append(sample)

        print(f'Num samples: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def normalize(self, midi_tuples):
        '''
        Normalizes MIDI tuples to [-1, 1] range
        midi_tuples: [melody length, 3]
        '''

        midi_tuples[:, 0] = 2 * (midi_tuples[:, 0] / self.MIDI_VAL_MAX) - 1
        midi_tuples[:, 1] = 2 * (midi_tuples[:, 1] / self.DUR_MAX) - 1
        midi_tuples[:, 2] = 2 * (midi_tuples[:, 2] / self.REST_MAX) - 1

        return midi_tuples

    def normalize2(self, midi_tuples):
        '''
        Normalizes MIDI tuples to [-1, 1] range
        midi_tuples: [melody length, 3]
        '''

        midi_tuples[:, 0] = ((midi_tuples[:, 0] / self.MIDI_VAL_MAX) - 0.5) / 0.5
        midi_tuples[:, 1] = ((midi_tuples[:, 1] / self.DUR_MAX) - 0.5) / 0.5
        midi_tuples[:, 2] = ((midi_tuples[:, 2] / self.REST_MAX) - 0.5) / 0.5

        return midi_tuples

    def denormalize(self, midi_tuples):
        '''
        Denormalizes MIDI tuples to original range
        midi_tuples: [melody length, 3]
        '''

        midi_tuples[:, 0] = (midi_tuples[:, 0] + 1) / 2 * self.MIDI_VAL_MAX
        midi_tuples[:, 1] = (midi_tuples[:, 1] + 1) / 2 * self.DUR_MAX
        midi_tuples[:, 2] = (midi_tuples[:, 2] + 1) / 2 * self.REST_MAX

        return midi_tuples

    def denormalize2(self, midi_tuples):
        '''
        Denormalizes MIDI tuples to original range
        midi_tuples: [melody length, 3]
        '''

        midi_tuples[:, 0] = (midi_tuples[:, 0] * 0.5 + 0.5) * self.MIDI_VAL_MAX
        midi_tuples[:, 1] = (midi_tuples[:, 1] * 0.5 + 0.5) * self.DUR_MAX
        midi_tuples[:, 2] = (midi_tuples[:, 2] * 0.5 + 0.5) * self.REST_MAX

        return midi_tuples

    def decenter(self, midi_tuples):
        '''
        De-mean centers the data 
        '''
        
        for channel,(mean, std) in enumerate(zip(self.means, self.stds)):
            midi_tuples[:, channel] = std*midi_tuples[:, channel] + mean

        return midi_tuples

    def deminmax(self, midi_tuples):
        '''
        De-mean centers the data 
        '''
        
        for channel,(ch_min, ch_max) in enumerate(self.ch_minmaxes):
            midi_tuples[:, channel] = (midi_tuples[:, channel] + 1) / 2 * (ch_max - ch_min) + ch_min

        return midi_tuples

    def discretize(self, sample):
        '''
        Discretizes the output of NNN to nearest value
        '''

        dist = np.inf
        authorized_values_pitch = range(127)
        authorized_values_duration = [0.25,  0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 16., 32.]
        authorized_values_rest = [0., 1., 2., 4., 8., 16., 32.]
        discretized_sample = np.zeros(shape=np.shape(sample))
        discretized_sample_arrays = []
        for i in range(len(sample)):
            for j in range(0, len(authorized_values_pitch)):
                if (sample[i][0] - authorized_values_pitch[j]) ** 2 < dist:
                    dist = (sample[i][0] - authorized_values_pitch[j]) ** 2
                    discretized_sample[i][0] = authorized_values_pitch[j]
            dist = np.inf
            for j in range(0, len(authorized_values_duration)):
                if (sample[i][1] - authorized_values_duration[j]) ** 2 < dist:
                    dist = (sample[i][1] - authorized_values_duration[j]) ** 2
                    discretized_sample[i][1] = authorized_values_duration[j]
            dist = np.inf
            for j in range(0, len(authorized_values_rest)):
                if (sample[i][2] - authorized_values_rest[j]) ** 2 < dist:
                    dist = (sample[i][2] - authorized_values_rest[j]) ** 2
                    discretized_sample[i][2] = authorized_values_rest[j]
            dist = np.inf
            discretized_sample_arrays.append(np.asarray(discretized_sample[i][:]))

        return np.array(discretized_sample_arrays).tolist()

    def create_midi_pattern_from_discretized_data(self, discretized_sample):
        new_midi = pretty_midi.PrettyMIDI()
        voice = pretty_midi.Instrument(1)  # It's here to change the used instruments !
        tempo = 120
        ActualTime = 0  # Time since the beginning of the song, in seconds
        for i in range(0,len(discretized_sample)):
            length = discretized_sample[i][1] * 60 / tempo  # Conversion Duration to Time
            if i < len(discretized_sample) - 1:
                gap = discretized_sample[i + 1][2] * 60 / tempo
            else:
                gap = 0  # The Last element doesn't have a gap
            note = pretty_midi.Note(velocity=100, pitch=int(discretized_sample[i][0]), start=ActualTime,
                                    end=ActualTime + length)
            voice.notes.append(note)
            ActualTime += length + gap  # Update of the time

        new_midi.instruments.append(voice)

        return new_midi