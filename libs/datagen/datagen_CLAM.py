from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from libs.utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

    df.to_csv(filename)
    print()


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 ignore=[],
                 label_col=None,
                 study_voting='max'):
        '''
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		'''
        self.label_dict = label_dict
        self.custom_test_ids = None
        self.num_classes = len(self.label_dict)
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col

        slide_data = pd.read_csv(csv_path)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

        ###shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data
        self.study_data_prep()
        self.cls_ids_prep()

    def study_data_prep(self):
        studies = np.unique(np.array(self.slide_data['case_id']))  # get unique patients
        study_labels = []
        study_grades = []

        # Assign label for study at first, using max
        for p in studies:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist() # indexex of slides in same study
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values                          # label of slides in same study
            label = label.max()  # get patient label (MIL convention)

            grade = self.slide_data['grade'][locations].values
            grade = grade.mean()

            study_grades.append(grade)
            study_labels.append(label)

        self.study_data = {'case_id': studies, 'label': np.array(study_labels), 'grade': np.array(study_grades)}

        self.study_partition()

    def study_partition(self):
        df = pd.DataFrame(self.study_data)
        N = len(self.study_data['label'])
        M = list(self.study_data['label']).count(self.label_dict[1.0])
        ratio = M / N
        
        print('Number of positive studies: {}'.format(M))
        print('Number of total studies: {}'.format(N))
        print('Verify and press any key to continue...')
        self.n_positive_studies = M
        self.n_negative_studies = N - M
        self.n_studies = N
        
#         if ratio > 0.6:  # too much positive, move lowest grade studies to negative
#             df_pos = df[df['label'] == self.label_dict[1.0]]
#             df_pos = df_pos.sort_values(by=['grade']).reset_index()
#             K = M - int(0.5 * N)
#             for i in range(K):
#                 s = df_pos.loc[i, 'case_id']
#                 self.study_data['label'][np.where(self.study_data['case_id'] == s)] = self.label_dict[0.0]

#         elif ratio < 0.4:  # too much negative, move highes grade studies to positive
            
#             df_neg = df[df['label'] == self.label_dict[0.0]]
#             df_neg = df_neg.sort_values(by=['grade'], ascending=False).reset_index()
#             K = int(0.5 * N) - M
#             for i in range(K):
#                 s = df_neg.loc[i, 'case_id']
#                 self.study_data['label'][np.where(self.study_data['case_id'] == s)] = self.label_dict[1.0]

    def cls_ids_prep(self):
        self.study_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.study_cls_ids[i] = np.where(self.study_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        # convert from MIL label
        data = data[['study_code', 'target', 'slide', 'grade']]
        data.rename(columns={'study_code': 'case_id', 'target': 'label', 'slide': 'slide_id'}, inplace=True)

        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

    def __len__(self):
        return len(self.study_data['case_id'])

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Study level; Number of samples registered in class %d: %d' % (i, self.study_cls_ids[i].shape[0]))
            print('Slide level; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': self.custom_test_ids
        }

        settings.update({'cls_ids': self.study_cls_ids, 'samples': len(self.study_data['case_id'])})

        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        slide_ids = [[] for i in range(len(ids))]

        for split in range(len(ids)):
            for idx in ids[split]:
                case_id = self.study_data['case_id'][idx]
                slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                slide_ids[split].extend(slide_indices)

        self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        
        split_fixed = []
        for a in split:
            if not isinstance(a, str):
                split_fixed.append(str(int(a)))
            else:
                split_fixed.append(a)
        print(split_fixed[0])

        if len(split_fixed) > 0:
            mask = self.slide_data['slide_id'].isin(split_fixed)
            df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
            print(df_slice.info())
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        else:
            split = None

        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        else:
            split = None

        return split

    def return_splits(self, from_id=True, csv_path=None):

        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)

            else:
                test_split = None

        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
            print(all_splits.info())
            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):

        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in
                     range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index,
                              columns=columns)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
                 data_dir,
                 **kwargs):

        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]

        if not self.use_h5:
            if self.data_dir:
                full_path = os.path.join(self.data_dir, '{}.pt'.format(slide_id))
                features = torch.load(full_path)
                return features, label

            else:
                return slide_id, label

        else:
            full_path = os.path.join(self.data_dir, '{}.h5'.format(slide_id))
            with h5py.File(full_path, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]

            features = torch.from_numpy(features)
            return features, label, coords


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.grade_avg_cls = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
            self.grade_avg_cls[i] = round(self.slide_data.loc[self.slide_cls_ids[i], 'grade'].mean(),3)

    def __len__(self):
        return len(self.slide_data)
