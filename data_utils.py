import numpy as np
from utils import LoadImage
import os
import random

class TrainsetLoader(object):
    def __init__(self, trainset_dir, upscale_factor, batch_size, n_iters, T_in):
        self.trainset_dir = trainset_dir
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.upscale_factor = upscale_factor
        self.T_in = T_in
        self.video_list = os.listdir(trainset_dir)

    def __getitem__(self, idx):
        L_batch = []
        H_batch = []
        for batch in range(self.batch_size):
            L_frames, H_frames = self.get_batch()
            L_batch.append(L_frames)
            H_batch.append(H_frames)
        return L_batch, H_batch

    def get_batch(self):
        idx_video = random.randint(0, self.video_list.__len__() - 1)
        idx_frame = random.randint(1, 100 - self.T_in)
        lr_dir = self.trainset_dir + '/' + self.video_list[idx_video]  + '/lr_x' + str(self.upscale_factor) + '_BI'
        hr_dir = self.trainset_dir + '/' + self.video_list[idx_video]  + '/hr'
        # read HR & LR frames
        L_frames = []
        for i in range(self.T_in):
            L_frames.append(LoadImage(lr_dir + '/lr' + str(idx_frame + i) + '.bmp'))
        H_frames = LoadImage(hr_dir + '/hr' + str(idx_frame + self.T_in // 2) + '.bmp')
        L_frames = np.asarray(L_frames)
        # pad L_frame
        L_frames_padded = np.lib.pad(L_frames, pad_width=((self.T_in // 2, self.T_in // 2), (0, 0), (0, 0), (0, 0)), mode='constant')
        #H_frames = np.asarray(H_frames[np.newaxis,np.newaxis,:,:,:])
        return L_frames, H_frames


    def __len__(self):
        return self.n_iters