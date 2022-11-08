import torch
import os
import random

import cv2
import numpy as np

from torch.utils.data import Dataset

import utils

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

        # Go through data splitting MIDI tuples and paired lyrics (syllable embeddings)
        self.samples = []
        for d in data:

            # Create sample
            midi_tuples = np.array(np.split(d[:MIDI_TUPLE_SIZE*SONG_LENGTH], SONG_LENGTH))
            syllable_embs = d[MIDI_TUPLE_SIZE*SONG_LENGTH:]
            sample = (midi_tuples, syllable_embs)
            self.samples.append(sample)

        print(f'Num samples: {len(self.samples)}')


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]