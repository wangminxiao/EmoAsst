from __future__ import annotations
import os
import os.path as osp
from typing import Literal, Optional
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from .tokenizer import tokenize
from model.pre_audio import MaskedAudioFeature

EMOTION_CLASS_NAMES = [
    'neutral',
    'surprise',
    'fear',
    'sadness',
    'joy',
    'disgust',
    'anger'
]

SENTIMENT_CLASS_NAMES = [
    'neutral',
    'positive',
    'negative'
]

class MELD(Dataset):
    def __init__(
        self,
        data_dir: str = '/mnt/ff1f01b3-85e2-407c-8f5d-cdcee532daa5/emodet_cache/MELD.Raw',
        split: Literal['train', 'dev', 'test'] = 'train',
        sampling_strategy: str = 'uniform',
        dense_sampling_interval: Optional[int] = 4,
        video_len: int = 8,
        target: Literal['utt_text', 'utt_token', 'emotion_idx', 'multimodal', 'multimodal_finetune'] = 'emotion_idx',
        audio_sample_rate: int = 16000,
    ):
        assert split in ['train', 'dev', 'test']
        assert sampling_strategy in ['random', 'uniform']
        
        super().__init__()
        # constants
        self.RESIZE_SIZE = 256
        self.CROP_SIZE = 224
        
        self.data_dir = data_dir
        self.split = split
        self.sampling_strategy = sampling_strategy
        self.dense_sampling_interval = dense_sampling_interval
        self.video_len = video_len
        self.target = target
        self.audio_sample_rate = audio_sample_rate

        self.index: pd.DataFrame
        self.all_human_boxes: dict
        self._create_index()
        
        # clip-style preprocesser
        self.preprocesser = T.Compose([
            T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.audioft_extract = MaskedAudioFeature(layer_id=11, device='cpu')
        
    def _create_index(self):
        # load .csv data
        if self.split in ['train', 'dev', 'test']:
            annotation_file_path = osp.join(self.data_dir, f'{self.split}_sent_emo.csv')
            self.index = pd.read_csv(annotation_file_path)
        else:
            raise NotImplementedError

        # some preprocessing
        def get_num_time(timestring):
            pt = datetime.strptime(timestring, '%H:%M:%S,%f')
            total_seconds = pt.microsecond * 1e-6 + pt.second + pt.minute * 60 + pt.hour * 3600
            return total_seconds

        self.index['Emotion'] = self.index['Emotion'].apply(lambda x: EMOTION_CLASS_NAMES.index(x))
        self.index['Sentiment'] = self.index['Sentiment'].apply(lambda x: SENTIMENT_CLASS_NAMES.index(x))

        # wmx add filter long audio
        self.index['StartTime'] = self.index['StartTime'].apply(lambda x: get_num_time(x))
        self.index['EndTime'] = self.index['EndTime'].apply(lambda x: get_num_time(x))
        self.index['Dur'] = self.index['EndTime'] - self.index['StartTime']
        self.index = self.index[self.index['Dur']<=7.5]
        
        # mannually remove corrupted samples
        if self.split == 'train':
            CORRUPTED_SAMPLES_INFO = [
                (1165, 125, 3)
            ]
        elif self.split == 'dev':
            CORRUPTED_SAMPLES_INFO = [
                (1084, 110, 7)
            ]
        elif self.split == 'test':
            CORRUPTED_SAMPLES_INFO = []
        for idx, dia_id, utt_id in CORRUPTED_SAMPLES_INFO:
            assert self.index.loc[idx, 'Dialogue_ID'] == dia_id
            assert self.index.loc[idx, 'Utterance_ID'] == utt_id
            self.index.drop(idx, inplace=True)
        self.index.reset_index(drop=True, inplace=True)
        
        
    def __len__(self):
        return self.index.shape[0]
    
    
    def __getitem__(self, i):
        dialogue_id = self.index.loc[i, 'Dialogue_ID']
        utterance_id = self.index.loc[i, 'Utterance_ID']
        clip_id = f'dia{dialogue_id}_utt{utterance_id}'
        clip_dir = osp.join(self.data_dir, f'{self.split}_splits/frames', clip_id)
        audio_path = osp.join(self.data_dir, f'{self.split}_splits/audio', clip_id + '.mp3')
        num_frames = len(os.listdir(clip_dir)) - 1

        # return another sample if the clip is too short when using non-uniform sampling
        if self.sampling_strategy != 'uniform' and num_frames < self.video_len:
            new_i = np.random.randint(self.index.shape[0])
            return self.__getitem__(new_i)
        
        #load audio features
        audio_feature, audio_mask = self.audioft_extract(audio_path, num_frames)

        # sampling
        start_frame, end_frame = 0, num_frames-1
        if self.sampling_strategy == 'uniform':
            sampled_frame_ids = np.linspace(start_frame, end_frame, self.video_len, dtype=int)
        elif self.sampling_strategy == 'random':
            sampled_frame_ids = sorted(np.random.choice(np.arange(start_frame, end_frame+1), self.video_len, replace=False))
        
        # load video frames
        frames = []
        for frame_id in sampled_frame_ids:
            frame_path = osp.join(clip_dir, '{:08d}'.format(frame_id + 1) + '.jpg')
            raw_frame = Image.open(frame_path).convert('RGB')

            resized_frame = F.resize(raw_frame, size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC)

            frame = F.center_crop(resized_frame, self.CROP_SIZE)
            frame = F.normalize(
                tensor=F.to_tensor(frame),
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            frames.append(frame)
        frames = torch.stack(frames, dim=0).float()

        # load text
        """
        Prompt_list = ['some one is speaking with a ' + EMOTION_CLASS_NAMES[self.index.loc[i, 'Emotion']] + ' emotion', #1
                       self.index.loc[i, 'Speaker'] + 'is speaking with a ' + EMOTION_CLASS_NAMES[
                           self.index.loc[i, 'Emotion']] + ' emotion', #2
                       'The speaker is ' + EMOTION_CLASS_NAMES[self.index.loc[i, 'Emotion']], #3
                       self.index.loc[i, 'Speaker'] + ' is feeling ' + EMOTION_CLASS_NAMES[self.index.loc[i, 'Emotion']], #4
                       'The speaker said ' + self.index.loc[i, 'Utterance'] + ', that means the speaker is ' + EMOTION_CLASS_NAMES[self.index.loc[i, 'Emotion']], #5
                       self.index.loc[i, 'Utterance'] + EMOTION_CLASS_NAMES[self.index.loc[i, 'Emotion']], #6

                       ]"""
        Prompt_list = ['some one is speaking with a ' + EMOTION_CLASS_NAMES[self.index.loc[i, 'Emotion']] + ' emotion']

        Utterance_text = random.choice(Prompt_list)

        if self.target == 'utt_text':
            target = Utterance_text
        elif self.target == 'utt_token':
            raw_text = Utterance_text
            target = tokenize(raw_text).squeeze()
        elif self.target == 'emotion_idx':
            target = self.index.loc[i, 'Emotion']
        elif self.target == 'multimodal':
            target = {
                'utt_text': Utterance_text,
                'utt_token': tokenize(Utterance_text).squeeze(),
                'emotion_idx': self.index.loc[i, 'Emotion'],
            }
        elif self.target == 'multimodal_finetune':
            target = {
                'utt_token': tokenize(self.index.loc[i, 'Utterance']).squeeze(),
                'emotion_idx': self.index.loc[i, 'Emotion'],
                'audio_wav': audio_feature,
            }
        else:
            raise NotImplementedError

        return frames, target
        
        
if __name__ == '__main__':
    pass
    