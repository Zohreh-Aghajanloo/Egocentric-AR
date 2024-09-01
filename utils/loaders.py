import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger

import random
import numpy as np

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                # model_features.shape: (5, 1024)
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################


        # for _ in range(self.num_clips):
        #     max_start_idx = record.num_frames[modality] - (self.num_frames_per_clip[modality] - 1) * self.stride
        #     start_idx = random.randint(0, max_start_idx - 1)
        #     clip = [start_idx + self.stride * i for i in range(self.num_frames_per_clip[modality])]
        #     indices += clip

        clip_length = record.num_frames[modality] // self.num_clips
        indices = []
        clip_overlap = False

        if clip_length < self.num_frames_per_clip[modality]:
            clip_length = record.num_frames[modality]
            clip_overlap = True
        
        for s in range(self.stride, 0, -1):
            if clip_length > (self.num_frames_per_clip[modality] - 1) * s:
                stride = s
                break

        if clip_overlap:
            max_start_idx = clip_length - (self.num_frames_per_clip[modality] - 1) * stride
            for _ in range(self.num_clips):
                start_idx = random.randint(0, max_start_idx - 1)
                samples = [start_idx + stride * i for i in range(self.num_frames_per_clip[modality])]
                indices += samples
            return(np.array(indices))

        elif self.dense_sampling[modality]:
            for offset in range(self.num_clips):
                max_start_idx = (offset + 1) * clip_length - (self.num_frames_per_clip[modality] - 1) * stride
                start_idx = random.randint(offset * clip_length, max_start_idx - 1)
                samples = [start_idx + stride * i for i in range(self.num_frames_per_clip[modality])]
                indices += samples
        else:
            space = clip_length // self.num_frames_per_clip[modality]
            samples = [offset * clip_length + space * i for i in range(self.num_frames_per_clip[modality]) for offset in range(self.num_clips)]
            indices += samples          

        return(np.array(indices))

    def _get_val_indices(self, record, modality):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################

        clip_length = record.num_frames[modality] // self.num_clips
        indices = []
        clip_overlap = False

        if clip_length < self.num_frames_per_clip[modality]:
            clip_length = record.num_frames[modality]
            clip_overlap = True
        
        for s in range(self.stride, 0, -1):
            if clip_length > (self.num_frames_per_clip[modality] - 1) * s:
                stride = s
                break

        if clip_overlap:
            max_start_idx = clip_length - (self.num_frames_per_clip[modality] - 1) * stride
            for _ in range(self.num_clips):
                start_idx = random.randint(0, max_start_idx - 1)
                samples = [start_idx + stride * i for i in range(self.num_frames_per_clip[modality])]
                indices += samples
            return(np.array(indices))

        if self.dense_sampling[modality]:
            start_idx = int(clip_length / 2 - (self.num_frames_per_clip[modality] - 1) * stride / 2)
            for offset in range(self.num_clips):
                samples = [offset * start_idx + stride * i for i in range(self.num_frames_per_clip[modality])]
                indices += samples
        else:
            space = clip_length // self.num_frames_per_clip[modality]
            samples = [offset * clip_length + space * i for i in range(self.num_frames_per_clip[modality]) for offset in range(self.num_clips)]
            indices += samples

        return(np.array(indices))

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)


class ActionNet(data.Dataset, ABC):
    def __init__(self, modalities, mode, dataset_conf, transform=None, load_feat=False, additional_info=False, **kwargs):
        self.modalities = modalities 
        self.mode = mode  
        self.dataset_conf = dataset_conf
        self.additional_info = additional_info

        if 'RGB' in self.modalities:
            pickle_name = "multimodal_" + self.mode + "_set.pkl"
        elif self.mode == "train":
            pickle_name = "train_set.pkl"
        else:
            pickle_name = "test_set.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {self.mode} with {len(self.list_file)} samples generated")
        self.record_list = [row for idx, row in self.list_file.iterrows()]
        self.transform = transform
        self.load_feat = load_feat

        if self.load_feat and 'RGB' in self.modalities:
            self.model_features = pd.DataFrame(pd.read_pickle(os.path.join("action-net", "saved_features", self.dataset_conf.features_name
                                                                           + "_" + self.mode + ".pkl"))['features'])
            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", left_on='index', right_index=True)

    def __getitem__(self, index):
        samples = {}
        record = self.record_list[index]
        label = record['label']

        if 'EMG' in self.modalities:
            left_arm = record['myo_left_readings']
            right_arm = record['myo_right_readings']
            samples['EMG'] = np.float32(np.concatenate((left_arm, right_arm), axis=1))

        if 'RGB' in self.modalities:            
            if self.load_feat:
                sample_row = self.model_features[self.model_features['index'] == record.name]
                assert len(sample_row) == 1
                samples['RGB'] = sample_row['featuresRGB'].values[0]
            else:
                segment_indices = self.get_frame_indices(record)
                img = self.get('RGB', segment_indices)
                samples['RGB'] = img

        if self.additional_info:
            return samples, label, record.name
        else:
            return samples, label
    
    def get_frame_indices(self, record, fps=30, first_frame=1655239114.183343):
        start_time = record['start'] - first_frame
        stop_time = record['stop'] - first_frame
        start_frame = int(start_time * fps) + 1
        stop_frame = int(stop_time * fps) + 1
        return(list(range(start_frame, stop_frame)))
    
    def get(self, modality, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            frame = self._load_data('RGB', p)
            images.extend(frame)
        process_data = self.transform[modality](images)
        return process_data
        
    def _load_data(self, modality, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB':
            try:
                img = Image.open(os.path.join(data_path, tmpl.format(idx))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path, "frame_*")))[-1].split("_")[-1].split(".")[0])
                if idx > max_idx_video:
                    img = Image.open(os.path.join(data_path, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")
    
    def __len__(self):
        return len(self.record_list)