import torch
import numpy as np
import pytorch_lightning as pl
from scipy.ndimage import zoom

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import GaussianNoiseTransform, NumpyToTensor, Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2

class ATLASDataLoader(DataLoader):

    def __init__(self, subjects, t1ws, batch_size):
        super().__init__(subjects, batch_size)
        self.t1ws = t1ws
        self.indices = list(range(len(subjects))) # auto shuffled

    def generate_train_batch(self):
        batch_indices = self.get_indices() # ensures we loop through data once per epoch

        # resample to 88x88x88 to get it working...
        data = self.t1ws[batch_indices][:,None,:,:,:].astype(np.float32)
        data = zoom(data, 2)

        seg = self.t1ws[batch_indices][:,None,:,:,:].astype(np.float32)
        seg = zoom(seg, 2)

        # return a dict of data and seg
        return {'data': data, 'seg': seg}



class ATLASDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, train_fraction=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_fraction = train_fraction

        # define the augmentation transforms
        transforms = []
        transforms.append(SpatialTransform_2(
                            patch_size=(176,176),
                            do_elastic_deform=True,
                            deformation_scale=(0, 0.2),
                            p_el_per_sample=0.2,
                            do_scale=True,
                            scale=(0.8, 1.2),
                            p_scale_per_sample=0.2,
                            do_rotation=True,
                            angle_x=(- 15/360. * 2 * np.pi, 15/360. * 2 * np.pi),
                            angle_y=(- 15/360. * 2 * np.pi, 15/360. * 2 * np.pi),
                            angle_z=(- 15/360. * 2 * np.pi, 15/360. * 2 * np.pi),
                            p_rot_per_sample=0.2,
                            random_crop=False,
                            border_mode_data='constant'
                ))
        transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.1))
        transforms.append(NumpyToTensor())
        self.transforms = Compose(transforms)

    def prepare_data(self):
        pass

    def setup(self, stage):
        # list the data files
        # split dataset into train/val/test
        sub_ids = np.loadtxt(self.data_dir + '/ATLAS_subject_ids.txt')
        sub_idxs = list(range(len(sub_ids)))

        # load the data in memory
        self.t1ws = np.load(self.data_dir + '/ATLAS_t1ws.npy')

        # select train_fraction subjects from sub_ids for training set
        # generate random (sub idx, plane, slice) tuples
        train_sub_idxs = list(np.random.choice(
                                sub_idxs,
                                int(len(sub_ids)*self.train_fraction),
                                replace=False
                                ))
        val_sub_idxs = list(set(sub_idxs).difference(set(train_sub_idxs)))

        self.train_dl = ATLASDataLoader(train_subjects, self.t1ws, self.batch_size)
        self.val_dl = ATLASDataLoader(val_subjects, self.t1ws, self.batch_size)

    def train_dataloader(self):
        return MultiThreadedAugmenter(self.train_dl, self.transforms, num_processes=2, num_cached_per_queue=2, seeds=None, pin_memory=True)

    def val_dataloader(self):
        return MultiThreadedAugmenter(self.val_dl, self.transforms, num_processes=2, num_cached_per_queue=2, seeds=None, pin_memory=True)

    def test_dataloader(self):
        pass
