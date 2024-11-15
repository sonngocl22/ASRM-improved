import json
import pickle
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from random import Random
from datetime import datetime as dt

from tqdm import tqdm


# settings
DNAMES = (
    'vwfs',
    'vwfscxt',
    'vwfsbids'
)
NUM_NEGATIVE_SAMPLES = 2000
USE_FILTER_OUT = False
MIN_ITEM_COUNT_PER_USER = 2
MIN_USER_COUNT_PER_ITEM = 1
ICONTEXT_COLUMNS = [
    'year',
    'month',
    'day',
    'dayofweek',
    'dayofyear',
    'week',
]


def parse_args():

    # constants
    tasks = {
        'prepare'
    }

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=tasks, help="task to do")
    parser.add_argument('--dname', type=str, choices=DNAMES, help="dataset name to do")
    parser.add_argument('--data_root', type=str, default='./data', help="data root dir")
    parser.add_argument('--raw_root', type=str, default='./raw', help="raw root dir")
    parser.add_argument('--force', default=False, action='store_true', help="force to do task (otherwise use cached)")
    parser.add_argument('--random_seed', type=int, default=12345, help="random seed")

    # postprocessing
    args = parser.parse_args()
    args.data_root = Path(args.data_root)
    args.raw_root = Path(args.raw_root)

    return args


def print_timedelta(tdo):
    print(f"({'.'.join(str(tdo).split('.')[:-1])})")


def append_icontext(df_rows):
    df_rows['dto'] = pd.to_datetime(df_rows['stamp'], unit='s')
    (
        df_rows['year'],
        df_rows['month'],
        df_rows['day'],
        df_rows['dayofweek'],
        df_rows['dayofyear'],
        df_rows['week'],
    ) = zip(*df_rows['dto'].map(lambda dto: (
        dto.year,
        dto.month,
        dto.day,
        dto.dayofweek,
        dto.dayofyear,
        dto.week,
    )))
    df_rows['year'] -= df_rows['year'].min()
    df_rows['year'] /= df_rows['year'].max()
    df_rows['year'] = df_rows['year'].fillna(0.0)
    df_rows['month'] /= 12
    df_rows['day'] /= 31
    df_rows['dayofweek'] /= 7
    df_rows['dayofyear'] /= 365
    df_rows['week'] /= 4
    df_rows = df_rows.drop(columns=['dto'])
    df_rows = df_rows[['uid', 'iid', 'stamp'] + ICONTEXT_COLUMNS]
    return df_rows


def do_general_preprocessing(args, df_rows):
    """
        Given `df_rows` with a right format, the rest will be done.

        Args:
            `args`: see `parse_args`.
            `df_rows`: a DataFrame with column of `(uid, iid, stamp, year, month, day, dayofweek, dayofyear, week)`.
    """
    print("do general preprocessing")

    data_dir = args.data_root / args.dname

    if USE_FILTER_OUT:

        # filter out tiny items
        print("- filter out tiny items")
        df_iid2ucount = df_rows.groupby('iid').size()
        survived_iids = df_iid2ucount.index[df_iid2ucount >= MIN_USER_COUNT_PER_ITEM]
        df_rows = df_rows[df_rows['iid'].isin(survived_iids)]

        # filter out tiny users
        print("- filter out tiny users")
        df_uid2icount = df_rows.groupby('uid').size()
        survived_uids = df_uid2icount.index[df_uid2icount >= MIN_ITEM_COUNT_PER_USER]
        df_rows = df_rows[df_rows['uid'].isin(survived_uids)]

    print("- map uid -> uindex", end=' ', flush=True)
    check = dt.now()
    ss_uids = df_rows.groupby('uid').size().sort_values(ascending=False)
    uids = list(ss_uids.index)
    uid2uindex = {uid: index for index, uid in enumerate(uids, start=1)}
    df_rows['uindex'] = df_rows['uid'].map(uid2uindex)
    df_rows = df_rows.drop(columns=['uid'])
    with open(data_dir / 'uid2uindex.pkl', 'wb') as fp:
        pickle.dump(uid2uindex, fp)
    print_timedelta(dt.now() - check)

    print("- map iid -> iindex", end=' ', flush=True)
    check = dt.now()
    ss_iids = df_rows.groupby('iid').size().sort_values(ascending=False)
    iids = list(ss_iids.index)
    iid2iindex = {iid: index for index, iid in enumerate(iids, start=1)}
    df_rows['iindex'] = df_rows['iid'].map(iid2iindex)
    df_rows = df_rows.drop(columns=['iid'])
    with open(data_dir / 'iid2iindex.pkl', 'wb') as fp:
        pickle.dump(iid2iindex, fp)
        
    print_timedelta(dt.now() - check)

    print("- save df_rows with icontext", end=' ', flush=True)
    check = dt.now()
    # Removed context features as there is non in VWFS currently
    #df_rows['icontext'] = df_rows[ICONTEXT_COLUMNS].apply(tuple, axis=1)
    #df_rows = df_rows.drop(columns=ICONTEXT_COLUMNS)
    #df_rows = df_rows[['uindex', 'iindex', 'stamp', 'icontext']]
    df_rows = df_rows[['uindex', 'iindex', 'stamp']]
    df_rows.to_pickle(data_dir / 'df_rows.pkl')
    print_timedelta(dt.now() - check)

    print("- split train, valid, test")
    uindex2urows_train = {}
    uindex2urows_valid = {}
    uindex2urows_test = {}
    for uindex in tqdm(list(uid2uindex.values()), desc="* splitting"):
        df_urows = df_rows[df_rows['uindex'] == uindex]
        urows = list(df_urows[['iindex', 'stamp']].itertuples(index=False, name=None))
        if len(urows) < 3:
            uindex2urows_train[uindex] = urows
        else:
            uindex2urows_train[uindex] = urows[:-2]
            uindex2urows_valid[uindex] = urows[-2:-1]
            uindex2urows_test[uindex] = urows[-1:]

    print("- save splits", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'uindex2urows_train.pkl', 'wb') as fp:
        pickle.dump(uindex2urows_train, fp)
    with open(data_dir / 'uindex2urows_valid.pkl', 'wb') as fp:
        pickle.dump(uindex2urows_valid, fp)
    with open(data_dir / 'uindex2urows_test.pkl', 'wb') as fp:
        pickle.dump(uindex2urows_test, fp)
    print_timedelta(dt.now() - check)


def do_general_random_negative_sampling(args):
    """
        The `ns_random.pkl` created here is a dict with `uindex` as a key and a list of `iindex` as a value.

        `ns_random` = {dict of `uindex` -> [list of `iindex`]}.
    """
    print("do general random negative sampling")

    print("- init dirs")
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load materials", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'df_rows.pkl', 'rb') as fp:
        df_rows = pickle.load(fp)
    with open(data_dir / 'uid2uindex.pkl', 'rb') as fp:
        uid2uindex = pickle.load(fp)
        num_users = len(uid2uindex)
    with open(data_dir / 'iid2iindex.pkl', 'rb') as fp:
        iid2iindex = pickle.load(fp)
        num_items = len(iid2iindex)
        iindex2iid = {iindex: iid for iid, iindex in iid2iindex.items()}
 
    with open('raw/ASRM/VWFS_ITEMSDATES.dat', 'rb') as fp:
        item2date = pickle.load(fp)
    with open('raw/ASRM/VWFS_DATEDICT.dat', 'rb') as fp:
        date2items = pickle.load(fp)
        
    with open(data_dir / 'uindex2urows_test.pkl', 'rb') as fp:
        uindex2urows_test = pickle.load(fp)     
        
    print_timedelta(dt.now() - check)

    print("- sample random negatives")
    print("Total num of items : ", num_items)
    print("Total num of users : ", num_users)
    ns = {}
    rng = Random(args.random_seed)
    for uindex in tqdm(list(range(1, num_users + 1)), desc="* sampling"):

        if uindex not in uindex2urows_test:
            continue
            
        test_iindex = uindex2urows_test[uindex][0][0]
        test_iid= iindex2iid[test_iindex]
        valid_item_date = item2date[test_iid]
        items_set = date2items[valid_item_date]    


        
        seen_iindices = set(df_rows[df_rows['uindex'] == uindex]['iindex'])
        sampled_iindices = set()
        
        # Selecting negative items that have occured in the same month as target item
        items_set_indexs = {iid2iindex[iid] for iid in items_set}        
        possible_items_iindices = list(set(items_set_indexs)-set(seen_iindices))
        sampled_iindices = list(np.random.choice(possible_items_iindices, NUM_NEGATIVE_SAMPLES))
            
        ns[uindex] = list(sampled_iindices)

    print("- save sampled random nagetives", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'ns_random.pkl', 'wb') as fp:
        pickle.dump(ns, fp)
    print_timedelta(dt.now() - check)


def do_create_ifeature_matrix(args):
    """
        Uses `iid2feature` and `iid2iindex` to create `ifeatures` matrix.

        0th row has 0-vector.

        Args:
            `args`: see `parse_args`.
    """
    print("do create ifeatures matrix")

    print("- init dirs")
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load materials", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'iid2iindex.pkl', 'rb') as fp:
        iid2iindex = pickle.load(fp)
        iindex2iid = {iindex: iid for iid, iindex in iid2iindex.items()}
    with open(data_dir / 'iid2ifeature.pkl', 'rb') as fp:
        iid2ifeature = pickle.load(fp)
    print_timedelta(dt.now() - check)

    print("- create ifeatures matrix", end=' ', flush=True)
    check = dt.now()
    ifeatures = []
    for iindex in range(1, len(iid2iindex) + 1):
        iid = iindex2iid[iindex]
        ifeature = iid2ifeature[iid]
        ifeatures.append(ifeature)
    ifeature_dim = len(ifeatures[0])
    ifeatures = [np.zeros(ifeature_dim)] + ifeatures
    ifeatures = np.array(ifeatures)
    print_timedelta(dt.now() - check)

    print("- save ifeatures matrix", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'ifeatures_wrnd.pkl', 'wb') as fp:
        pickle.dump(ifeatures, fp)
    print_timedelta(dt.now() - check)


def preprocess_asrm(args, ifeature_fname, icontext_fname, rows_fname):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / 'ASRM'
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load ifeature data", end=' ', flush=True)
    check = dt.now()
    if not (data_dir / 'iid2ifeature.pkl').is_file():
        iid2ifeature = {}
        with open(raw_dir / ifeature_fname, 'rb') as fp:
            index2ifeature = pickle.load(fp)
            for iid, ifeature in enumerate(index2ifeature, start=1):
                iid = int(iid)
                iid2ifeature[iid] = ifeature
        with open(data_dir / 'iid2ifeature.pkl', 'wb') as fp:
            pickle.dump(iid2ifeature, fp)
    print_timedelta(dt.now() - check)

    # print("- load icontext data", end=' ', flush=True)
    # check = dt.now()
    # with open(raw_dir / icontext_fname, 'rb') as fp:
    #     uidiid2icontext = pickle.load(fp)
    # print_timedelta(dt.now() - check)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        df_rows = pd.read_csv(raw_dir / rows_fname, dtype={
            0: int,
            1: int,
            2: int,
        }, delim_whitespace=True, header=None)
        df_rows.columns = ['uid', 'iid', 'stamp']
        df_rows = df_rows.sort_values('stamp', ascending=True)
        df_rows.to_parquet(raw_dir / fname)
    print_timedelta(dt.now() - check)

    print("- make raw df", end=' ', flush=True)
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)

    print("- append icontext", end=' ', flush=True)
    check = dt.now()
    rows = []
    for uid, iid, stamp in df_rows.itertuples(index=False, name=None):
        # icontext = uidiid2icontext[(uid, iid)]
        #Removing context features 
        #rows.append([uid, iid, stamp] + list(icontext))
        rows.append([uid, iid, stamp])
    df_rows = pd.DataFrame(rows)
    #Removing context features 
    #df_rows.columns = ['uid', 'iid', 'stamp'] + ICONTEXT_COLUMNS
    df_rows.columns = ['uid', 'iid', 'stamp']
    print_timedelta(dt.now() - check)

    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_create_ifeature_matrix(args)

    print("done")
    print()

def preprocess_bids(args, rows_fname):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / 'ASRM'
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        df_rows = pd.read_csv(raw_dir / rows_fname, dtype={
            0: int,
            1: int,
            2: int,
        }, delim_whitespace=True, header=None)
        df_rows.columns = ['uid', 'iid', 'stamp']
        df_rows = df_rows.sort_values('stamp', ascending=True)
        df_rows.to_parquet(raw_dir / fname)

    print("- make raw df", end=' ', flush=True)
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)
    
    do_bids_preprocessing(args, df_rows)


def do_bids_preprocessing(args, df_rows):

    data_dir = args.data_root / args.dname
    vwfs_dir = args.data_root / 'vwfs'

    if USE_FILTER_OUT:

        # filter out tiny items
        print("- filter out tiny items")
        df_iid2ucount = df_rows.groupby('iid').size()
        survived_iids = df_iid2ucount.index[df_iid2ucount >= MIN_USER_COUNT_PER_ITEM]
        df_rows = df_rows[df_rows['iid'].isin(survived_iids)]

        # filter out tiny users
        print("- filter out tiny users")
        df_uid2icount = df_rows.groupby('uid').size()
        survived_uids = df_uid2icount.index[df_uid2icount >= MIN_ITEM_COUNT_PER_USER]
        df_rows = df_rows[df_rows['uid'].isin(survived_uids)]

    # loading the sales indexes to filter out unknown users/items from bids
    with open(vwfs_dir / 'iid2iindex.pkl', 'rb') as fp:
        sales_iid2iindex = pickle.load(fp)
    with open(vwfs_dir / 'uid2uindex.pkl', 'rb') as fp:
        sales_uid2uindex = pickle.load(fp)

    sales_uidset = set(sales_uid2uindex.keys())
    sales_iidset = set(sales_iid2iindex.keys())

    df_rows['uindex'] = df_rows['uid'].map(sales_uid2uindex)
    df_rows = df_rows.drop(columns=['uid'])

    df_rows['iindex'] = df_rows['iid'].map(sales_iid2iindex)
    df_rows = df_rows.drop(columns=['iid'])

    df_rows = df_rows[['uindex', 'iindex', 'stamp']]
    filtered_df_rows = df_rows[
        df_rows['uindex'].isin(sales_uidset) &
        df_rows['iindex'].isin(sales_iidset)
    ]

    filtered_df_rows = filtered_df_rows.astype(int)
    filtered_df_rows.to_pickle(data_dir / 'df_rows.pkl')

    

    print("- saving uindex2urows_bids")
    uindex2urows_bids = {}
    for uindex in tqdm(list(sales_uid2uindex.values()), desc="* splitting"):
        df_urows = filtered_df_rows[filtered_df_rows['uindex'] == uindex]
        urows = list(df_urows[['iindex', 'stamp']].itertuples(index=False, name=None))
        uindex2urows_bids[uindex] = urows

    with open(data_dir / 'uindex2urows_bids.pkl', 'wb') as fp:
        pickle.dump(uindex2urows_bids, fp)

    
def task_prepare_vwfs(args):
    preprocess_asrm(
        args,
        ifeature_fname='vwfs_feat_wrnd.dat',
        icontext_fname='CXTDictSasRec_VWFS_NoContext.dat',
        rows_fname='vwfs_cxt.txt'
    )
def task_prepare_vwfscxt(args):
    preprocess_asrm(
        args,
        ifeature_fname='vwfs_feat_wrnd.dat',
        icontext_fname='CXTDictSasRec_VWFS.dat',
        rows_fname='vwfs_cxt.txt'
    )
# for bids
def task_prepare_vwfsbids(args):
    preprocess_bids(
        args,
        rows_fname='bids_full.txt'
    )

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'prepare':
        globals()[f'task_{args.task}_{args.dname}'](args)
    else:
        globals()[f'task_{args.task}'](args)
