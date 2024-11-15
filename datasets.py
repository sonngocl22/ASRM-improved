import os
import pickle
import random

import numpy as np
import pandas as pd

from typing import Optional

from torch import (
    Tensor,
    LongTensor,
)
from torch.utils.data import Dataset


__all__ = (
    'LWPContrastiveTrainDataset',
    'EvalDataset'
)

class LWPContrastiveTrainDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 sequence_len: int,
                 random_cut_prob: float = 1.0,
                 replace_user_prob: float = 0.0,
                 replace_item_prob: float = 0.02,
                 train_num_negatives: int = 100,
                 random_seed: Optional[int] = None,
                 mainrel: float = 1.0
                 ):

        # params
        self.name = name
        self.sequence_len = sequence_len
        self.random_cut_prob = random_cut_prob
        self.replace_user_prob = replace_user_prob
        self.replace_item_prob = replace_item_prob
        self.train_num_negatives = train_num_negatives
        self.random_seed = random_seed
        ##
        self.data_root = data_root
        self.mainrel = mainrel

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # load data
        with open(os.path.join(self.data_root, name, 'uid2uindex.pkl'), 'rb') as fp:
            self.uid2uindex = pickle.load(fp)
            self.uindex2uid = {uindex: uid for uid, uindex in self.uid2uindex.items()}
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.iindex2iid = {iindex: iid for iid, iindex in self.iid2iindex.items()}
#            Sam modification to vectirize negative sampling
            self.iid2iindex_series = pd.Series(self.iid2iindex)
        with open(os.path.join(self.data_root, name, 'uindex2urows_train.pkl'), 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)

        ## loading vwfs sales date data for new negative samples
        if self.name == 'vwfs' or self.name == 'vwfscxt':
            with open('raw/ASRM/VWFS_ITEMSDATES.dat', 'rb') as fp:
                self.item2date = pickle.load(fp)
            with open('raw/ASRM/VWFS_DATEDICT.dat', 'rb') as fp:
                self.date2items = pickle.load(fp)
        ## loading vwfs bids date data for new negative samples
        if self.mainrel < 1.0:
            with open('raw/ASRM/VWFS_BIDITEMSDATES.dat', 'rb') as fp:
                self.item2date_b = pickle.load(fp)
            with open('raw/ASRM/VWFS_BIDDATEDICT.dat', 'rb') as fp:
                self.date2items_b = pickle.load(fp)
            with open(os.path.join(self.data_root, 'vwfsbids', 'uindex2urows_bids.pkl'), 'rb') as fp:
                self.uindex2urows_bids = pickle.load(fp)


        # settle down
        self.uindices = []
        self.iindexset_train = set()
        self.uindex2iindexset = {}
        for uindex, urows in self.uindex2urows_train.items():
            iindexset_user = set()

            #modifications:
            if self.name == 'vwfs':
                for iindex, _ in urows:
                    iindexset_user.add(iindex)
                    self.iindexset_train.add(iindex)
            else:
                for iindex, _, _ in urows:
                    iindexset_user.add(iindex)
                    self.iindexset_train.add(iindex)

            self.uindex2iindexset[uindex] = iindexset_user
            if len(urows) < 2:
                continue
            self.uindices.append(uindex)
        self.iindices_train = list(self.iindexset_train)
        self.num_items = len(self.iid2iindex)
        self.stamp_min = 9999999999
        self.stamp_max = 0
        for _, urows in self.uindex2urows_train.items():
            #modifications:
            if self.name == 'vwfs':
                for _, stamp in urows:
                    if stamp > self.stamp_max:
                        self.stamp_max = stamp
                    if stamp < self.stamp_min:
                        self.stamp_min = stamp
            else:
                for _, stamp, _ in urows:
                    if stamp > self.stamp_max:
                        self.stamp_max = stamp
                    if stamp < self.stamp_min:
                        self.stamp_min = stamp
        self.stamp_interval = self.stamp_max - self.stamp_min

        # tokens
        self.padding_token = 0

        # icontext info
        #modification : Removing context features for vwfs dataset
        if self.name == 'vwfs':
            self.icontext_dim = 0
        else:
            _, _, sample_icontext = urows[0]
            self.icontext_dim = len(sample_icontext)
        self.padding_icontext = np.zeros(self.icontext_dim)

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # data point
        uindex = self.uindices[index]
        urows = self.uindex2urows_train[uindex]
        iindexset_point = set(self.uindex2iindexset[uindex])
        num_iindices_train = len(self.iindices_train)

        if self.mainrel < 1.0:
            urows_bids = self.uindex2urows_bids[uindex]

        # data driven regularization: replace user (see SSE-PT)
        if self.rng.random() < self.replace_user_prob:
            sampled_index = self.rng.randrange(0, len(self.uindices))
            uindex = self.uindices[sampled_index]

        # long sequence random cut (see SSE-PT++)
        if self.rng.random() < self.random_cut_prob:
            urows = urows[:self.rng.randint(2, len(urows))]

        # last as positive
        if self.name == 'vwfs':
            positive_token, positive_stamp = urows[-1]
        else:
            positive_token, positive_stamp, positive_icontext = urows[-1]
        extract_tokens = [positive_token]

        if self.mainrel < 1.0:
            df_urows_bids = pd.DataFrame(urows_bids, columns=['iindex', 'stamp'])
            filtered_df_urows_bids = df_urows_bids[df_urows_bids['stamp'] <= positive_stamp]
            # sample the bid item with the closest timestamp to the sales item
            if not filtered_df_urows_bids.empty:
                max_timestamp = filtered_df_urows_bids['stamp'].max()
                candidate_bids = filtered_df_urows_bids[filtered_df_urows_bids['stamp'] == max_timestamp]
                positive_token_bid, positive_stamp_bid = candidate_bids.sample(n=1).iloc[0][['iindex', 'stamp']]
                extract_tokens_bid = [positive_token_bid]
            else:
                extract_tokens_bid = [positive_token]
                

        # bake profile
        profile_tokens = []
        profile_stamps = []
        profile_icontexts = []

        if self.name == 'vwfs':
            for profile_iindex, profile_stamp in urows[:-1][-self.sequence_len:]:

                # data driven regularization: replace item (see SSE)
                if self.rng.random() < self.replace_item_prob:
                    sampled_index = self.rng.randrange(0, num_iindices_train)
                    profile_iindex = self.iindices_train[sampled_index]
                    iindexset_point.add(profile_iindex)

                # add item
                profile_tokens.append(profile_iindex)
                profile_stamps.append(profile_stamp)
                profile_icontexts.append([])
        else:
            for profile_iindex, profile_stamp, profile_icontext in urows[:-1][-self.sequence_len:]:

                # data driven regularization: replace item (see SSE)
                if self.rng.random() < self.replace_item_prob:
                    sampled_index = self.rng.randrange(0, num_iindices_train)
                    profile_iindex = self.iindices_train[sampled_index]
                    iindexset_point.add(profile_iindex)

                # add item
                profile_tokens.append(profile_iindex)
                profile_stamps.append(profile_stamp)
                profile_icontexts.append(profile_icontext)

        # add paddings
        if self.name == 'vwfs':
            _, padding_stamp = urows[0]
        else:
            _, padding_stamp, _ = urows[0]
        padding_len = self.sequence_len - len(profile_tokens)
        profile_tokens = [self.padding_token] * padding_len + profile_tokens
        profile_stamps = [padding_stamp] * padding_len + profile_stamps
        profile_icontexts = [self.padding_icontext] * padding_len + profile_icontexts

        # sample negatives
        ############ changing negative sampling for train set to be difficult
        ### further modifications to vectorize negative sampling
        negative_tokens = set()
        if self.name == 'vwfs':
            # picking the items that have occured in the same month as target item
            pos_iid = self.iid2iindex[positive_token]
            pos_item_date = self.item2date[pos_iid]
            neg_items_set = self.date2items[pos_item_date]
            neg_items_list = np.array(neg_items_set)
            items_set_indices = self.iid2iindex_series[neg_items_list].values
            iindexset_point_array = np.array(list(iindexset_point))
            valid_negatives = np.setdiff1d(items_set_indices, iindexset_point_array)
            num_needed_negatives = self.train_num_negatives
            if len(valid_negatives) >= num_needed_negatives:
                selected_negatives = np.random.choice(valid_negatives, num_needed_negatives, replace=False)
            else:
                selected_negatives = valid_negatives  # In case there are fewer valid negatives available

            # Convert selected_negatives to a set and update negative_tokens
            negative_tokens.update(selected_negatives)
        else:
            while len(negative_tokens) < self.train_num_negatives:
                while True:
                    sample_index = self.rng.randrange(0, num_iindices_train)
                    negative_iindex = self.iindices_train[sample_index]
                    if negative_iindex not in iindexset_point and negative_iindex not in negative_tokens:
                        break
                negative_tokens.add(negative_iindex)

        negative_tokens = list(negative_tokens)
        extract_tokens.extend(negative_tokens)
        if self.mainrel < 1.0:
            extract_tokens_bid.extend(negative_tokens)

        # fill extract
        extract_stamps = []
        extract_icontexts = []
        for _ in extract_tokens:
            extract_stamps.append(positive_stamp)
            #modifications:
            if self.name == 'vwfs':
                extract_icontexts.append([])
            else:
                extract_icontexts.append(positive_icontext)

        # return tensorized data point
        # option to return both sales and bids
        sales_item = {
            'uindex': uindex,
            'profile_tokens': LongTensor(profile_tokens), # (b x L) -> 100 (self.sequence_len) profile sequence tokens
            'profile_stamps': LongTensor(profile_stamps),
            'profile_icontexts': Tensor(np.array(profile_icontexts)),
            'extract_tokens': LongTensor(extract_tokens), # (b x C) -> 1 positive and 100 (self.train_num_negatives) negative tokens
            'extract_stamps': LongTensor(extract_stamps),
            'extract_icontexts': Tensor(np.array(extract_icontexts)),
            'label': 0,
        }
        if self.mainrel < 1.0:
            bids_item = {
                'uindex': uindex,
                'profile_tokens': LongTensor(profile_tokens), # (b x L) -> 100 (self.sequence_len) profile sequence tokens
                'profile_stamps': LongTensor(profile_stamps),
                'profile_icontexts': Tensor(np.array(profile_icontexts)),
                'extract_tokens': LongTensor(extract_tokens_bid), # (b x C) -> 1 positive and 100 (self.train_num_negatives) negative tokens
                'extract_stamps': LongTensor(extract_stamps),
                'extract_icontexts': Tensor(np.array(extract_icontexts)),
                'label': 0,
            }
            return sales_item, bids_item
        else:
            return sales_item


class EvalDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 target: str,  # 'valid', 'test'
                 sequence_len: int,
                 valid_num_negatives: int = 100,
                 random_seed: Optional[int] = None
                 ):

        # params
        self.name = name
        self.target = target
        self.sequence_len = sequence_len
        self.valid_num_negatives = valid_num_negatives
        self.random_seed = random_seed
        ##
        self.data_root = data_root

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # load data
        with open(os.path.join(self.data_root, name, 'uid2uindex.pkl'), 'rb') as fp:
            self.uid2uindex = pickle.load(fp)
            self.uindex2uid = {uindex: uid for uid, uindex in self.uid2uindex.items()}
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.iindex2iid = {iindex: iid for iid, iindex in self.iid2iindex.items()}
        with open(os.path.join(self.data_root, name, 'uindex2urows_train.pkl'), 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)
            self.iindexset_train = set()
            #modifications:
            for uindex, urows in self.uindex2urows_train.items():
                if self.name == 'vwfs':
                    for iindex, _ in urows:
                        self.iindexset_train.add(iindex)
                else:
                    for iindex, _, _ in urows:
                        self.iindexset_train.add(iindex)
        with open(os.path.join(self.data_root, name, 'uindex2urows_valid.pkl'), 'rb') as fp:
            self.uindex2urows_valid = pickle.load(fp)
            self.iindexset_valid = set()
            for uindex, urows in self.uindex2urows_valid.items():
                #modifications:
                if self.name == 'vwfs':
                    for iindex, _ in urows:
                        self.iindexset_valid.add(iindex)
                else:
                    for iindex, _, _ in urows:
                        self.iindexset_valid.add(iindex)
        with open(os.path.join(self.data_root, name, 'uindex2urows_test.pkl'), 'rb') as fp:
            self.uindex2urows_test = pickle.load(fp)
            self.uindex2aiindexset_test = {}
            for uindex, urows in self.uindex2urows_test.items():
                aiindexset = set()
                #modifications:
                if self.name == 'vwfs':
                    for iindex, _ in urows:
                        aiindexset.add(iindex)
                else:
                    for iindex, _, _ in urows:
                        aiindexset.add(iindex)
                self.uindex2aiindexset_test[uindex] = aiindexset
        with open(os.path.join(self.data_root, name, 'ns_random.pkl'), 'rb') as fp:
            self.uindex2negatives_test = pickle.load(fp)

        # settle down
        if target == 'valid':
            self.uindices = []
            for uindex in self.uindex2urows_valid:
                if uindex in self.uindex2urows_train:
                    self.uindices.append(uindex)
        elif target == 'test':
            self.uindices = []
            for uindex in self.uindex2aiindexset_test:
                if uindex not in self.uindex2urows_train and uindex not in self.uindex2urows_valid:
                    continue
                self.uindices.append(uindex)
        self.iindexset_known = set()
        self.uindex2iindexset = {}
        for uindex, urows in self.uindex2urows_train.items():
            iindexset_user = set()
            #modifications:
            if self.name == 'vwfs':
                for iindex, _ in urows:
                    iindexset_user.add(iindex)
                    self.iindexset_known.add(iindex)
            else:
                for iindex, _, _ in urows:
                    iindexset_user.add(iindex)
                    self.iindexset_known.add(iindex)
            self.uindex2iindexset[uindex] = iindexset_user
        for uindex, urows in self.uindex2urows_valid.items():
            iindexset_user = set()
            #modifications:
            if self.name == 'vwfs':
                for iindex, _ in urows:
                    iindexset_user.add(iindex)
                    self.iindexset_known.add(iindex)
            else:
                for iindex, _, _ in urows:
                    iindexset_user.add(iindex)
                    self.iindexset_known.add(iindex)
            if uindex not in self.uindex2iindexset:
                self.uindex2iindexset[uindex] = set()
            self.uindex2iindexset[uindex] |= iindexset_user
        self.iindices_known = list(self.iindexset_known)
        self.num_items = len(self.iid2iindex)

        # tokens
        self.padding_token = 0

        # icontext info
        #modification : Removing context features for vwfs dataset
        if self.name == 'vwfs':
            self.icontext_dim = 0
        else:
            _, _, sample_icontext = urows[0]
            self.icontext_dim = len(sample_icontext)
        self.padding_icontext = np.zeros(self.icontext_dim)

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # get data point
        uindex = self.uindices[index]
        urows_train = self.uindex2urows_train.get(uindex, [])
        urows_valid = self.uindex2urows_valid.get(uindex, [])
        urows_test = self.uindex2urows_test.get(uindex, [])

        # prepare rows
        if self.target == 'valid':
            urows_known = urows_train
            urows_eval = urows_valid
        elif self.target == 'test':
            urows_known = urows_train  + urows_valid
            urows_eval = urows_test

        # get eval row
        #modifications:
        if self.name == 'vwfs':
            answer_iindex, answer_stamp = urows_eval[0]
        else:
            answer_iindex, answer_stamp, answer_icontext = urows_eval[0]
        extract_tokens = [answer_iindex]

        # bake profile
        profile_tokens = []
        profile_stamps = []
        profile_icontexts = []
        #Son modifications:
        if self.name == 'vwfs':
            for profile_iindex, profile_stamp in urows_known[-self.sequence_len:]:
                profile_tokens.append(profile_iindex)
                profile_stamps.append(profile_stamp)
                profile_icontexts.append([])
        else:
            for profile_iindex, profile_stamp, profile_icontext in urows_known[-self.sequence_len:]:
                profile_tokens.append(profile_iindex)
                profile_stamps.append(profile_stamp)
                profile_icontexts.append(profile_icontext)

        # add paddings
        #modifications:
        if self.name == 'vwfs':
            _, padding_stamp = urows_known[0]
        else:
            _, padding_stamp, _ = urows_known[0]
        padding_len = self.sequence_len - len(profile_tokens)
        profile_tokens = [self.padding_token] * padding_len + profile_tokens
        profile_stamps = [padding_stamp] * padding_len + profile_stamps
        profile_icontexts = [self.padding_icontext] * padding_len + profile_icontexts

        # sample negatives
        if self.target == 'valid':
            negative_tokens = self.uindex2negatives_test[uindex]
        elif self.target == 'test':
            # print(f'uindex for test target: {uindex}')
            negative_tokens = self.uindex2negatives_test[uindex]
        extract_tokens.extend(negative_tokens)

        # bake extract
        extract_stamps = []
        extract_icontexts = []
        for _ in extract_tokens:
            extract_stamps.append(answer_stamp)
            if self.name == 'vwfs':
                pass
            else:
                extract_icontexts.append(answer_icontext)
        labels = [1] + [0] * (len(extract_tokens) - 1)

        # return tensorized data point
        return {
            'uindex': uindex,
            'profile_tokens': LongTensor(profile_tokens),
            'profile_stamps': Tensor(np.array(profile_stamps)),
            'profile_icontexts': Tensor(np.array(profile_icontexts)),
            'extract_tokens': LongTensor(extract_tokens),
            'extract_stamps': Tensor(np.array(extract_stamps)),
            'extract_icontexts': Tensor(np.array(extract_icontexts)),
            'labels': LongTensor(labels),
        }

