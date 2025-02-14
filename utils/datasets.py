import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import logging
from sklearn.model_selection import train_test_split
import json


dtypes_avazu = {
    'click': np.int8,
    # 'hour':np.int16,
    'C1':np.int8,
    'banner_pos':np.int8,
    'site_id':np.int16,
    'site_domain':np.int16,
    'site_category':np.int8,
    'app_id':np.int16,
    'app_domain':np.int16,
    'app_category':np.int8,
    'device_id':np.int32,
    'device_ip':np.int32,
    'device_model':np.int16,
    'device_type':np.int8,
    'device_conn_type':np.int8,
    'C14':np.int16,
    'C15':np.int8,
    'C16':np.int8,
    'C17':np.int16,
    'C18':np.int8,
    'C19':np.int8,
    'C20':np.int16,
    'C21':np.int8
}


class MyDataset(Dataset):
    def __init__(self, data, labels, indices):
        self.data = data
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.indices[idx]

    def get_random_indices(self, topk_num : int):
        return self.indices[torch.randperm(len(self))[:topk_num]]


def read_dataset(dataset_name, data_path, batch_size, shuffle, num_workers, use_fields=None, machine_learning_method=False):
    if machine_learning_method:
        if dataset_name == 'avazu':
            return read_avazu_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'criteo':
            return read_criteo_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'movielens-1m':
            return read_movielens1m_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'aliccp':
            return read_aliccp_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'kuairandpure':
            return read_kuairandpure_ml(data_path, batch_size, shuffle)
    elif not machine_learning_method:
        if dataset_name == 'avazu':
            return read_avazu(data_path, batch_size, shuffle, num_workers, use_fields)
        elif dataset_name == 'criteo':
            return read_criteo(data_path, batch_size, shuffle, num_workers, use_fields)
        elif dataset_name == 'movielens-1m':
            return read_movielens1m(data_path, batch_size, shuffle, num_workers, use_fields)
        elif dataset_name == 'aliccp':
            return read_aliccp(data_path, batch_size, shuffle, num_workers, use_fields)
        elif dataset_name == 'kuairandpure':
            return read_kuairandpure(data_path, batch_size, shuffle, num_workers, use_fields)

def read_dataset_curfull(dataset_name, data_path, batch_size, shuffle, num_workers, use_fields=None, machine_learning_method=False, indices_to_drop=None, args=None):

    if machine_learning_method:
        if dataset_name == 'avazu':
            return read_avazu_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'criteo':
            return read_criteo_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'movielens-1m':
            return read_movielens1m_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'aliccp':
            return read_aliccp_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'kuairandpure':
            return read_kuairandpure_ml(data_path, batch_size, shuffle)
    elif not machine_learning_method:
        if dataset_name == 'avazu':
            return read_avazu_curfull3(args, data_path, indices_to_drop)
        elif dataset_name == 'criteo':
            return read_criteo_curfull3(args, data_path, indices_to_drop)
        elif dataset_name == 'movielens-1m':
            return read_movielens1m_curfull3(args, data_path, indices_to_drop)
        elif dataset_name == 'aliccp':
            return read_aliccp_curfull3(args, data_path, indices_to_drop)
        elif dataset_name == 'kuairandpure':
            return read_kuairandpure_curfull3(args, data_path, indices_to_drop)

def read_split_dataset(dataset_name, data_path, batch_size, shuffle, num_workers, use_fields=None, machine_learning_method=False, indices_to_drop=[], args=None):
    if machine_learning_method:
        if dataset_name == 'avazu':
            return read_avazu_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'criteo':
            return read_criteo_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'movielens-1m':
            return read_movielens1m_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'aliccp':
            return read_aliccp_ml(data_path, batch_size, shuffle)
        elif dataset_name == 'kuairandpure':
            return read_kuairandpure_ml(data_path, batch_size, shuffle)
    elif not machine_learning_method:
        if dataset_name == 'avazu':
            return split_avazu(args, data_path, indices_to_drop)
        elif dataset_name == 'criteo':
            return split_criteo(args, data_path, indices_to_drop)
        elif dataset_name == 'movielens-1m':
            return split_movielens1m(args, data_path, indices_to_drop)
        elif dataset_name == 'aliccp':
            return split_aliccp(args, data_path, indices_to_drop)
        elif dataset_name == 'kuairandpure':
            return split_kuairandpure(args, data_path, indices_to_drop)
        
        
def read_avazu(data_path, batch_size, shuffle, num_workers, use_fields=None):
    dtypes = {
        'click': np.int8,
        'hour':np.int16,
        'C1':np.int8,
        'banner_pos':np.int8,
        'site_id':np.int16,
        'site_domain':np.int16,
        'site_category':np.int8,
        'app_id':np.int16,
        'app_domain':np.int16,
        'app_category':np.int8,
        'device_id':np.int32,
        'device_ip':np.int32,
        'device_model':np.int16,
        'device_type':np.int8,
        'device_conn_type':np.int8,
        'C14':np.int16,
        'C15':np.int8,
        'C16':np.int8,
        'C17':np.int16,
        'C18':np.int8,
        'C19':np.int8,
        'C20':np.int16,
        'C21':np.int8
    }
    print('start reading avazu...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'avazu/preprocessed_avazu.csv'), dtype = dtypes)
    else:
        df = pd.read_csv(os.path.join(data_path, 'avazu/preprocessed_avazu.csv'), dtype = dtypes, usecols=list(use_fields)+['click'])
    print('finish reading avazu.')
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['click']]
    unique_values = [df[col].max()+1 for col in features]
    label = 'click'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
    train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return features, label, train_dataloader, val_dataloader, test_dataloader, unique_values


def read_avazu_ml(data_path, batch_size, shuffle, use_fields=None):

    print('start reading avazu...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'avazu/preprocessed_avazu.csv'), dtype = dtypes_avazu)
#         df.drop(columns=['item_id:token'], inplace=True)
    else:
        df = pd.read_csv(os.path.join(data_path, 'avazu/preprocessed_avazu.csv'), dtype = dtypes_avazu, usecols=list(use_fields)+['click'])
    print('finish reading avazu.')
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['click']]
    unique_values = [df[col].max()+1 for col in features]
    label = 'click'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    return features, unique_values, (train_x, train_y, val_x, val_y, test_x, test_y)

def read_criteo(data_path, batch_size, shuffle, num_workers, use_fields=None):
    dtypes = {
        '0': np.int8,
        '1': np.int8,
        '2': np.int8,
        '3': np.int8,
        '4': np.int8,
        '5': np.int16,
        '6': np.int16,
        '7': np.int8,
        '8': np.int8,
        '9': np.int8,
        '10': np.int8,
        '11': np.int8,
        '12': np.int8,
        '13': np.int8,
        '14': np.int16,
        '15': np.int16,
        '16': np.int32,
        '17': np.int32,
        '18': np.int16,
        '19': np.int8,
        '20': np.int16,
        '21': np.int16,
        '22': np.int8,
        '23': np.int32,
        '24': np.int16,
        '25': np.int32,
        '26': np.int16,
        '27': np.int8,
        '28': np.int16,
        '29': np.int32,
        '30': np.int8,
        '31': np.int16,
        '32': np.int16,
        '33': np.int8,
        '34': np.int32,
        '35': np.int8,
        '36': np.int8,
        '37': np.int32,
        '38': np.int8,
        '39': np.int32
    }
    print('start reading criteo...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'criteo/preprocessed_criteo.csv'), dtype = dtypes)
#         df.drop(columns=['index:float'], inplace=True)
    else:
        df = pd.read_csv(os.path.join(data_path, 'criteo/preprocessed_criteo.csv'), dtype = dtypes, usecols=list(use_fields)+['0'])
    print('finish reading criteo.')
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['0']]
    unique_values = [df[col].max()+1 for col in features]
    label = '0'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
    train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return features, label, train_dataloader, val_dataloader, test_dataloader, unique_values






def read_criteo_ml(data_path, batch_size, shuffle, use_fields=None):
    dtypes = {
        '0': np.int8,
        '1': np.int8,
        '2': np.int8,
        '3': np.int8,
        '4': np.int8,
        '5': np.int16,
        '6': np.int16,
        '7': np.int8,
        '8': np.int8,
        '9': np.int8,
        '10': np.int8,
        '11': np.int8,
        '12': np.int8,
        '13': np.int8,
        '14': np.int16,
        '15': np.int16,
        '16': np.int32,
        '17': np.int32,
        '18': np.int16,
        '19': np.int8,
        '20': np.int16,
        '21': np.int16,
        '22': np.int8,
        '23': np.int32,
        '24': np.int16,
        '25': np.int32,
        '26': np.int16,
        '27': np.int8,
        '28': np.int16,
        '29': np.int32,
        '30': np.int8,
        '31': np.int16,
        '32': np.int16,
        '33': np.int8,
        '34': np.int32,
        '35': np.int8,
        '36': np.int8,
        '37': np.int32,
        '38': np.int8,
        '39': np.int32
    }
    print('start reading criteo...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'criteo/preprocessed_criteo.csv'), dtype = dtypes)
#         df.drop(columns=['index:float'], inplace=True)
    else:
        df = pd.read_csv(os.path.join(data_path, 'criteo/preprocessed_criteo.csv'), dtype = dtypes, usecols=list(use_fields)+['0'])
    print('finish reading criteo.')
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['0']]
    unique_values = [df[col].max()+1 for col in features]
    label = '0'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    return features, unique_values, (train_x, train_y, val_x, val_y, test_x, test_y)

def read_movielens1m(data_path, batch_size, shuffle, num_workers, use_fields=None):
    print('start reading movielens 1m...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'movielens-1m/ml-1m.csv'))
    else:
        df = pd.read_csv(os.path.join(data_path, 'movielens-1m/ml-1m.csv'), usecols=list(use_fields)+['rating'])
    print('finish reading movielens 1m.')
    df['rating'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
    df = df.sample(frac=1, random_state=43) # shuffle
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['rating']]
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    unique_values = [df[col].max()+1 for col in features]
    label = 'rating'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
    train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return features, label, train_dataloader, val_dataloader, test_dataloader, unique_values

def read_movielens1m_ml(data_path, batch_size, shuffle, use_fields=None):
    print('start reading movielens 1m...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'movielens-1m/ml-1m.csv'))
#         df.drop(columns=['item_id:token'], inplace=True)
    else:
        df = pd.read_csv(os.path.join(data_path, 'movielens-1m/ml-1m.csv'), usecols=list(use_fields)+['rating'])
    print('finish reading movielens 1m.')
    df['rating'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
    df = df.sample(frac=1, random_state=43) # shuffle
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['rating']]
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    unique_values = [df[col].max()+1 for col in features]
    label = 'rating'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    return features, unique_values, (train_x, train_y, val_x, val_y, test_x, test_y)

def read_aliccp(data_path, batch_size, shuffle, num_workers, use_fields=None):
    print('start reading aliccp...')
    data_type = {'click':np.int8, 'purchase': np.int8, '101':np.int32, '121':np.uint8, '122':np.uint8, '124':np.uint8, '125':np.uint8, '126':np.uint8, '127':np.uint8, '128':np.uint8, '129':np.uint8, '205':np.int32, '206':np.int16, '207':np.int32, '210':np.int32, '216':np.int32, '508':np.int16, '509':np.int32, '702':np.int32, '853':np.int32, '301':np.int8, '109_14':np.int16, '110_14':np.int32, '127_14':np.int32, '150_14':np.int32, 'D109_14': np.float16, 'D110_14': np.float16, 'D127_14': np.float16, 'D150_14': np.float16, 'D508': np.float16, 'D509': np.float16, 'D702': np.float16, 'D853': np.float16}
    if use_fields is None:
        df1 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_train.csv'), dtype=data_type)
        df2 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_val.csv'), dtype=data_type)
        df3 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_test.csv'), dtype=data_type)
        df = pd.concat([df1, df2, df3])
    else:
        df1 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_train.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df2 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_val.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df3 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_test.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df = pd.concat([df1, df2, df3])
    print('finish reading aliccp.')
    # df = df.sample(frac=1) # shuffle
    train_idx = int(df.shape[0] * 0.5)
    val_idx = int(df.shape[0] * 0.75)
    features = []
    for f in df.columns:
        if f not in ['click','purchase'] and f[:1] != 'D':
            features.append(f)
    if '301' in features:
        df['301'] = df['301'] - 1
    unique_values = [df[col].max()+1 for col in features]
    label = 'click'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
    train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return features, label, train_dataloader, val_dataloader, test_dataloader, unique_values

def read_aliccp_ml(data_path, batch_size, shuffle, use_fields=None):
    print('start reading aliccp...')
    data_type = {'click':np.int8, 'purchase': np.int8, '101':np.int32, '121':np.uint8, '122':np.uint8, '124':np.uint8, '125':np.uint8, '126':np.uint8, '127':np.uint8, '128':np.uint8, '129':np.uint8, '205':np.int32, '206':np.int16, '207':np.int32, '210':np.int32, '216':np.int32, '508':np.int16, '509':np.int32, '702':np.int32, '853':np.int32, '301':np.int8, '109_14':np.int16, '110_14':np.int32, '127_14':np.int32, '150_14':np.int32, 'D109_14': np.float16, 'D110_14': np.float16, 'D127_14': np.float16, 'D150_14': np.float16, 'D508': np.float16, 'D509': np.float16, 'D702': np.float16, 'D853': np.float16}
    if use_fields is None:
        df1 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_train.csv'), dtype=data_type)
        df2 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_val.csv'), dtype=data_type)
        df3 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_test.csv'), dtype=data_type)
        df = pd.concat([df1, df2, df3])
    else:
        df1 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_train.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df2 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_val.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df3 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_test.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df = pd.concat([df1, df2, df3])
    print('finish reading aliccp.')
    # df = df.sample(frac=1) # shuffle
    train_idx = int(df.shape[0] * 0.5)
    val_idx = int(df.shape[0] * 0.75)
    features = []
    for f in df.columns:
        if f not in ['click','purchase'] and f[:1] != 'D':
            features.append(f)
    df['301'] = df['301'] - 1
    unique_values = [df[col].max()+1 for col in features]
    label = 'click'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    return features, unique_values, (train_x, train_y, val_x, val_y, test_x, test_y)

def read_kuairandpure(data_path, batch_size, shuffle, num_workers, use_fields=None):
    print('start reading kuairand-pure ...')
    file_path = 'kuairand-pure/preprocessed-kuairandpure.csv'
    # file_path = 'kuairand-pure/preprocessed-kuairandpure-100.csv'  # for debug only
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, file_path))
    else:
        df = pd.read_csv(os.path.join(data_path, file_path), usecols=list(use_fields)+['is_click'])
    print('finish reading kuairand-pure.')
    df = df.sample(frac=1, random_state=43) # shuffle, 0.145 for scarce data experiments, 1 for normal experiment
    train_idx = int(df.shape[0] * 0.7) # 0.7 0.241
    val_idx = int(df.shape[0] * 0.9) # 0.9 0.3103
    features = [f for f in df.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
    unique_values = [df[col].max()+1 for col in features]
    label = 'is_click'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
    train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return features, label, train_dataloader, val_dataloader, test_dataloader, unique_values

def read_kuairandpure_ml(data_path, batch_size, shuffle, use_fields=None):
    print('start reading kuairand-pure ...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'kuairand-pure/preprocessed-kuairandpure.csv'))
    else:
        df = pd.read_csv(os.path.join(data_path, 'kuairand-pure/preprocessed-kuairandpure.csv'), usecols=list(use_fields)+['is_click'])
    print('finish reading kuairand-pure.')
    df = df.sample(frac=1, random_state=43) # shuffle, 0.145 for scarce data experiments, 1 for normal experiment
    train_idx = int(df.shape[0] * 0.7) # 0.7 0.241
    val_idx = int(df.shape[0] * 0.9) # 0.9 0.3103
    features = [f for f in df.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
    unique_values = [df[col].max()+1 for col in features]
    label = 'is_click'
    train_x, val_x, test_x = df[features][:train_idx], df[features][train_idx:val_idx], df[features][val_idx:]
    train_y, val_y, test_y = df[label][:train_idx], df[label][train_idx:val_idx], df[label][val_idx:]
    return features, unique_values, (train_x, train_y, val_x, val_y, test_x, test_y)


def split_avazu(args, data_path, indices_to_drop=set()):

    print(f'start reading {args.dataset}...')
    if args.read_mode == 'common':
        file_path = 'avazu/data_wash/train.csv'
        file_path_val = 'avazu/data_wash/val.csv'
    elif args.read_mode == 'sw':
        assert args.sw_previous_len is not None
        assert args.sw_t is not None
        assert args.sw_read_idx is not None
        read_root = f'avazu/slide_window/previous_len_{args.sw_previous_len}-{args.sw_t}/{args.sw_read_idx}'
        file_path = os.path.join(read_root, 'train.csv')
        file_path_val = os.path.join(read_root, 'test.csv')
        
    df_full = pd.read_csv(os.path.join(data_path, file_path), dtype = dtypes_avazu)
    df_val = pd.read_csv(os.path.join(data_path, file_path_val), dtype = dtypes_avazu)
    
    print(f'finish reading {args.dataset}.')
    if indices_to_drop:
        df_full.drop(index=list(indices_to_drop), inplace=True)
        
    features = [f for f in df_full.columns if f not in ['click']]
    label = 'click'
    
    return split_dataset(args, df_full, df_val, features, label)


def split_aliccp(args, data_path, indices_to_drop=set()):
    data_type = {'click':np.int8, 'purchase': np.int8, '101':np.int32, '121':np.uint8, '122':np.uint8, '124':np.uint8, '125':np.uint8, '126':np.uint8, '127':np.uint8, '128':np.uint8, '129':np.uint8, '205':np.int32, '206':np.int16, '207':np.int32, '210':np.int32, '216':np.int32, '508':np.int16, '509':np.int32, '702':np.int32, '853':np.int32, '301':np.int8, '109_14':np.int16, '110_14':np.int32, '127_14':np.int32, '150_14':np.int32, 'D109_14': np.float16, 'D110_14': np.float16, 'D127_14': np.float16, 'D150_14': np.float16, 'D508': np.float16, 'D509': np.float16, 'D702': np.float16, 'D853': np.float16}
    print(f'start reading {args.dataset} ...')
    file_path = 'aliccp/ali_ccp_train.csv'
    file_path_val = 'aliccp/ali_ccp_val.csv'
    
    df_full = pd.read_csv(os.path.join(data_path, file_path), dtype=data_type)
    df_val = pd.read_csv(os.path.join(data_path, file_path_val), dtype=data_type)

    print(f'finish reading {args.dataset}.')    
    
    if indices_to_drop:
        df_full.drop(index=list(indices_to_drop), inplace=True)
    
    features = []
    for f in df_full.columns:
        if f not in ['click','purchase'] and f[:1] != 'D':
            features.append(f)
    if '301' in features:
        df_full['301'] = df_full['301'] - 1
    label = 'click'
    
    return split_dataset(args, df_full, df_val, features, label)


def split_movielens1m(args, data_path, indices_to_drop=set()):
    print(f'start reading {args.dataset} ...')
    file_path = 'movielens-1m/data_wash/train.csv'
    file_path_val = 'movielens-1m/data_wash/val.csv'
    
    df_full = pd.read_csv(os.path.join(data_path, file_path))
    df_val = pd.read_csv(os.path.join(data_path, file_path_val))
    for df in (df_full, df_val):
        df['rating'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
        
    print(f'finish reading {args.dataset}.')    
    
    if indices_to_drop:
        df_full.drop(index=list(indices_to_drop), inplace=True)
    
    features = [f for f in df_full.columns if f not in ['rating']]
    def _process(_df):
        for feature in features:
            le = LabelEncoder()
            _df[feature] = le.fit_transform(_df[feature])
        return _df
    df_val, df_full = map(_process, (df_val, df_full))
    label = 'rating'
    return split_dataset(args, df_full, df_val, features, label)

     
def split_kuairandpure(args, data_path, indices_to_drop=set()):
    print(f'start reading {args.dataset} ...')
    if args.read_mode == 'common':
        file_path = 'kuairand-pure/data_wash/train.csv'
        file_path_val = 'kuairand-pure/data_wash/val.csv'
        
    elif args.read_mode == 'sw':
        assert args.sw_previous_len is not None
        assert args.sw_t is not None
        assert args.sw_read_idx is not None
        read_root = f'kuairand-pure/slide_window/previous_len_{args.sw_previous_len}-{args.sw_t}/{args.sw_read_idx}'
        file_path = os.path.join(read_root, 'train.csv')
        file_path_val = os.path.join(read_root, 'test.csv')
        
    df_full = pd.read_csv(os.path.join(data_path, file_path))
    df_val = pd.read_csv(os.path.join(data_path, file_path_val))
    
    print(f'finish reading {args.dataset}.')    
    
    if indices_to_drop:
        df_full.drop(index=list(indices_to_drop), inplace=True)
    
    df_full = df_full.sample(frac=1, random_state=43) # shuffle
    features = [f for f in df_full.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
    label = 'is_click'
    
    return split_dataset(args, df_full, df_val, features, label)


def split_criteo(args, data_path, indices_to_drop=set()):
    dtypes = {
        '0': np.int8,
        '1': np.int8,
        '2': np.int8,
        '3': np.int8,
        '4': np.int8,
        '5': np.int16,
        '6': np.int16,
        '7': np.int8,
        '8': np.int8,
        '9': np.int8,
        '10': np.int8,
        '11': np.int8,
        '12': np.int8,
        '13': np.int8,
        '14': np.int16,
        '15': np.int16,
        '16': np.int32,
        '17': np.int32,
        '18': np.int16,
        '19': np.int8,
        '20': np.int16,
        '21': np.int16,
        '22': np.int8,
        '23': np.int32,
        '24': np.int16,
        '25': np.int32,
        '26': np.int16,
        '27': np.int8,
        '28': np.int16,
        '29': np.int32,
        '30': np.int8,
        '31': np.int16,
        '32': np.int16,
        '33': np.int8,
        '34': np.int32,
        '35': np.int8,
        '36': np.int8,
        '37': np.int32,
        '38': np.int8,
        '39': np.int32
    }
    print(f'start reading {args.dataset}...')
    if args is not None and args.mode_debug:
        file_path = 'criteo/data_wash/train-1000.csv'  # for debug
        file_path_val = 'criteo/data_wash/val.csv'
    else:
        file_path = 'criteo/data_wash/train.csv'
        file_path_val = 'criteo/data_wash/val.csv'
        
    df_full = pd.read_csv(os.path.join(data_path, file_path), dtype = dtypes)
    df_val = pd.read_csv(os.path.join(data_path, file_path_val), dtype = dtypes)

    print(f'finish reading {args.dataset}.')
    
    if indices_to_drop:
        df_full.drop(index=list(indices_to_drop), inplace=True)
    features = [f for f in df_full.columns if f not in ['0']]
    label = '0'
    
    return split_dataset(args, df_full, df_val, features, label)
    

def split_dataset(args, df_full, df_val, features, label):
    split_idx = int(df_full.shape[0] * 0.5)
    x_split_1, x_split_2 = df_full[features][:split_idx], df_full[features][split_idx:]
    y_split_1, y_split_2 = df_full[label][:split_idx], df_full[label][split_idx:]
    index_1, index_2 = df_full.index[:split_idx], df_full.index[split_idx:]
    
    x_val, y_val, index_val = df_val[features], df_val[label], df_val.index
    x_val, y_val, index_val = torch.tensor(x_val.values, dtype=torch.long), torch.tensor(y_val.values, dtype=torch.long), torch.tensor(index_val.values, dtype=torch.long)
    
    x_split_1, x_split_2 = torch.tensor(x_split_1.values, dtype=torch.long), torch.tensor(x_split_2.values, dtype=torch.long)
    y_split_1, y_split_2 = torch.tensor(y_split_1.values, dtype=torch.long), torch.tensor(y_split_2.values, dtype=torch.long)
    index_1, index_2 = torch.tensor(index_1.values, dtype=torch.int64), torch.tensor(index_2.values, dtype=torch.int64)

    dataset_1 = MyDataset(x_split_1, y_split_1, index_1)
    dataset_2 = MyDataset(x_split_2, y_split_2, index_2)
    dataset_val = MyDataset(x_val, y_val, index_val)
    
    dataloader_1 = DataLoader(dataset_1, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers)
    dataloader_2 = DataLoader(dataset_2, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers)
    return features, label, dataloader_1, dataloader_2, dataloader_val
    

def read_criteo_curfull3(args, data_path, indices_to_drop=set()):
    dtypes = {
        '0': np.int8,
        '1': np.int8,
        '2': np.int8,
        '3': np.int8,
        '4': np.int8,
        '5': np.int16,
        '6': np.int16,
        '7': np.int8,
        '8': np.int8,
        '9': np.int8,
        '10': np.int8,
        '11': np.int8,
        '12': np.int8,
        '13': np.int8,
        '14': np.int16,
        '15': np.int16,
        '16': np.int32,
        '17': np.int32,
        '18': np.int16,
        '19': np.int8,
        '20': np.int16,
        '21': np.int16,
        '22': np.int8,
        '23': np.int32,
        '24': np.int16,
        '25': np.int32,
        '26': np.int16,
        '27': np.int8,
        '28': np.int16,
        '29': np.int32,
        '30': np.int8,
        '31': np.int16,
        '32': np.int16,
        '33': np.int8,
        '34': np.int32,
        '35': np.int8,
        '36': np.int8,
        '37': np.int32,
        '38': np.int8,
        '39': np.int32
    }
    
    print(f'start reading {args.dataset}...')
    if args is not None and args.mode_debug:
        train_path = 'criteo/data_wash/train-1000.csv'
        val_path = 'criteo/data_wash/val-1000.csv'
    else:
        train_path = 'criteo/data_wash/train.csv'
        val_path = 'criteo/data_wash/val.csv'
    test_path = 'criteo/data_wash/test.csv'
    full_path = 'criteo/preprocessed_criteo.csv'
    
    df_train, df_val, df_test, df_full = map(lambda path: pd.read_csv(os.path.join(data_path, path), dtype = dtypes), (train_path, val_path, test_path, full_path))
    
    print(f'finish reading {args.dataset}.')
    
    if indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)

    features = [f for f in df_full.columns if f not in ['0']]
    label = '0'
    
    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


def read_aliccp_curfull3(args, data_path, indices_to_drop=set()):
    print(f'start reading {args.dataset} ...')
    data_type = {'click':np.int8, 'purchase': np.int8, '101':np.int32, '121':np.uint8, '122':np.uint8, '124':np.uint8, '125':np.uint8, '126':np.uint8, '127':np.uint8, '128':np.uint8, '129':np.uint8, '205':np.int32, '206':np.int16, '207':np.int32, '210':np.int32, '216':np.int32, '508':np.int16, '509':np.int32, '702':np.int32, '853':np.int32, '301':np.int8, '109_14':np.int16, '110_14':np.int32, '127_14':np.int32, '150_14':np.int32, 'D109_14': np.float16, 'D110_14': np.float16, 'D127_14': np.float16, 'D150_14': np.float16, 'D508': np.float16, 'D509': np.float16, 'D702': np.float16, 'D853': np.float16}
    
    train_path = 'aliccp/ali_ccp_train.csv'
    val_path = 'aliccp/ali_ccp_val.csv'
    test_path = 'aliccp/ali_ccp_test.csv'
    
    df_train, df_val, df_test = map(lambda path: pd.read_csv(os.path.join(data_path, path), dtype=data_type), (train_path, val_path, test_path))
    df_full = pd.concat([df_train, df_val, df_test])
    
    print(f'finish reading {args.dataset}.')
    if indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)
        
    features = []
    for f in df_full.columns:
        if f not in ['click','purchase'] and f[:1] != 'D':
            features.append(f)
    if '301' in features:
        df_full['301'] = df_full['301'] - 1
    label = 'click'
    
    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


def read_avazu_curfull3(args, data_path, indices_to_drop):

    print(f'start reading {args.dataset}...')
    if args.read_mode == 'common':
        train_path = 'avazu/data_wash/train.csv'
        val_path = 'avazu/data_wash/val.csv'
        test_path = 'avazu/data_wash/test.csv'
        full_path = 'avazu/preprocessed_avazu.csv'
        df_train, df_val, df_test, df_full = map(lambda path: pd.read_csv(os.path.join(data_path, path), dtype = dtypes_avazu), (train_path, val_path, test_path, full_path))
    elif args.read_mode == 'sw':
        assert args.sw_previous_len is not None
        assert args.sw_t is not None
        assert args.sw_read_idx is not None
        read_root = f'avazu/slide_window/previous_len_{args.sw_previous_len}-{args.sw_t}/{args.sw_read_idx}'
        read_root = os.path.join(data_path, read_root)
        train_path = os.path.join(read_root, 'train.csv')
        val_path = os.path.join(read_root, 'test.csv')
        test_path = os.path.join(read_root, 'test.csv')
        df_train, df_val, df_test = map(lambda path: pd.read_csv(path), (train_path, val_path, test_path))
        df_full = pd.concat([df_train, df_test])

    print(f'finish reading {args.dataset}.')
    
    if indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)

    features = [f for f in df_full.columns if f not in ['click']]
    label = 'click'

    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


def read_movielens1m_curfull3(args, data_path, indices_to_drop=set()):
    print(f'start reading {args.dataset} ...')
    
    train_path = 'movielens-1m/data_wash/train.csv'
    val_path = 'movielens-1m/data_wash/val.csv'
    test_path = 'movielens-1m/data_wash/test.csv'
    full_path = 'movielens-1m/ml-1m.csv'
    
    df_train, df_val, df_test, df_full = map(lambda path: pd.read_csv(os.path.join(data_path, path)), (train_path, val_path, test_path, full_path))
    for df in (df_train, df_val, df_test, df_full):
        df['rating'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
    
    print(f'finish reading {args.dataset}.')
    if indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)
        
    features = [f for f in df_full.columns if f not in ['rating']]
    def _process(_df):
        for feature in features:
            le = LabelEncoder()
            _df[feature] = le.fit_transform(_df[feature])
        return _df
    df_train, df_val, df_test, df_full = map(_process, (df_train, df_val, df_test, df_full))
    label = 'rating'
    
    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


def read_kuairandpure_curfull3(args, data_path, indices_to_drop=set()):
    print(f'start reading {args.dataset} ...')
    
    if args.read_mode == 'common':
        train_path = 'kuairand-pure/data_wash/train.csv'
        val_path = 'kuairand-pure/data_wash/val.csv'
        test_path = 'kuairand-pure/data_wash/test.csv'
        full_path = 'kuairand-pure/preprocessed-kuairandpure.csv'
        df_train, df_val, df_test, df_full = map(lambda path: pd.read_csv(os.path.join(data_path, path)), (train_path, val_path, test_path, full_path))
    elif args.read_mode == 'sw':
        assert args.sw_previous_len is not None
        assert args.sw_t is not None
        assert args.sw_read_idx is not None
        read_root = f'kuairand-pure/slide_window/previous_len_{args.sw_previous_len}-{args.sw_t}/{args.sw_read_idx}'
        read_root = os.path.join(data_path, read_root)
        train_path = os.path.join(read_root, 'train.csv')
        val_path = os.path.join(read_root, 'test.csv')
        test_path = os.path.join(read_root, 'test.csv')
    
        df_train, df_val, df_test = map(lambda path: pd.read_csv(path), (train_path, val_path, test_path))
        df_full = pd.concat([df_train, df_test])
    
    print(f'finish reading {args.dataset}.')
    if indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)
        
    features = [f for f in df_full.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
    label = 'is_click'
    
    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


def read_curfull3(args, df_train, df_val, df_test, df_full, features, label):
    unique_values = [df_full[col].max()+1 for col in features]
    
    train_x, val_x, test_x = df_train[features], df_val[features], df_test[features]
    train_y, val_y, test_y = df_train[label], df_val[label], df_test[label]

    train_idx, val_idx, test_idx = map(lambda df: df.index, (df_train, df_val, df_test))
    train_idx, val_idx, test_idx = map(lambda x: torch.tensor(x.values, dtype=torch.int64), (train_idx, val_idx, test_idx))
    train_x, val_x, test_x, train_y, val_y, test_y = map(lambda x: torch.tensor(x.values, dtype=torch.long), (train_x, val_x, test_x, train_y, val_y, test_y))
    
    dataset_train = MyDataset(train_x, train_y, train_idx)
    dataset_val = MyDataset(val_x, val_y, val_idx)
    dataset_test = MyDataset(test_x, test_y, test_idx)
    
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers)
    val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers)
    return features, label, train_dataloader, val_dataloader, test_dataloader, unique_values


def get_dataset_from_df(df, features, label):
    x = torch.tensor(df[features].values, dtype=torch.long)
    y = torch.tensor(df[label].values, dtype=torch.long)
    idx = torch.tensor(df.index.values, dtype=torch.int64)
    dataset = MyDataset(x, y, idx)
    return dataset
    

def get_dataloader_from_df(args, df, features, label):
    dataset = get_dataset_from_df(df, features, label)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers)
    return dataloader

    
def read_one_day_full_v02(args, subset_root, indices_to_drop=set()):
    ''' for slide window v02 '''
    train_file_name = args.train_file_name if args.train_file_name is not None else 'train'
    val_file_name = args.val_file_name if args.val_file_name is not None else 'test'
    test_file_name = args.test_file_name if args.test_file_name is not None else 'test'
    train_path = os.path.join(subset_root, f'{train_file_name}.csv')
    val_path = os.path.join(subset_root, f'{val_file_name}.csv')
    test_path = os.path.join(subset_root, f'{test_file_name}.csv')
    logging.info(f"train_path: {train_path}")
    logging.info(f"val_path: {val_path}")
    logging.info(f"test_path: {test_path}")
    
    df_train, df_val, df_test = map(lambda path: pd.read_csv(path), (train_path, val_path, test_path))
    df_full = pd.concat([df_train, df_test])
    
    if args.mode == 'ncv' and indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)
    
    if args.dataset == 'kuairandpure':
        features = [f for f in df_full.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
        label = 'is_click'
    else:
        raise ValueError(args.dataset)
        
    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


def read_one_day_full_v03(args, subset_root, indices_to_drop=set()):
    ''' for slide window '''
    train_file_name = args.train_file_name if args.train_file_name is not None else 'subset_train'
    val_file_name = args.val_file_name if args.val_file_name is not None else 'subset_test'
    test_file_name = args.test_file_name if args.test_file_name is not None else 'next_day'
    train_path = os.path.join(subset_root, f'{train_file_name}.csv')
    val_path = os.path.join(subset_root, f'{val_file_name}.csv')
    test_path = os.path.join(subset_root, f'{test_file_name}.csv')
    logging.info(f"train_path: {train_path}")
    logging.info(f"val_path: {val_path}")
    logging.info(f"test_path: {test_path}")
    
    df_train, df_val, df_test = map(lambda path: pd.read_csv(path), (train_path, val_path, test_path))
    if args.get_full == 'train':
        df_full = df_train
    elif args.get_full == 'train_test':
        df_full = pd.concat([df_train, df_test])
    elif args.get_full == 'train_val_test':
        df_full = pd.concat([df_train, df_val, df_test])
    else:
        raise ValueError(args.get_full)
    
    if args.mode == 'ncv' and indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)
    
    if args.dataset == 'kuairandpure':
        features = [f for f in df_full.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
        label = 'is_click'
    else:
        raise ValueError(args.dataset)
        
    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


def read_one_day(args, subset_root, indices_to_drop=set()):
    if args.read_mode == 'test_only':
        return read_one_day_test_only(args, subset_root, indices_to_drop)
    elif args.read_mode == 'all':
        return read_one_day_full_v04(args, subset_root, indices_to_drop)


def read_one_day_full_v04(args, subset_root, indices_to_drop=set()):
    ''' for slide window '''
    ''' full加载全部的 kuairand '''
    train_file_name = args.train_file_name if args.train_file_name is not None else 'subset_train'
    val_file_name = args.val_file_name if args.val_file_name is not None else 'subset_test'
    test_file_name = args.test_file_name if args.test_file_name is not None else 'next_day'
    train_path = os.path.join(subset_root, f'{train_file_name}.csv')
    val_path = os.path.join(subset_root, f'{val_file_name}.csv')
    test_path = os.path.join(subset_root, f'{test_file_name}.csv')
    if args.dataset == 'kuairandpure':
        full_path = 'data/kuairand-pure/preprocessed-kuairandpure.csv'
    else:
        raise ValueError(args.dataset)
    logging.info(f"train_path: {train_path}")
    logging.info(f"val_path: {val_path}")
    logging.info(f"test_path: {test_path}")
    logging.info(f"full_path: {full_path}")
    
    df_train, df_val, df_test, df_full = map(lambda path: pd.read_csv(path), (train_path, val_path, test_path, full_path))
    
    if args.mode == 'ncv' and indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)
    
    if args.dataset == 'kuairandpure':
        features = [f for f in df_full.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
        label = 'is_click'
    else:
        raise ValueError(args.dataset)
        
    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


def read_one_day_test_only(args, subset_root, indices_to_drop=set()):
    test_file_name = args.test_file_name if args.test_file_name is not None else 'next_day'
    test_path = os.path.join(subset_root, f'{test_file_name}.csv')
    logging.info(f"test_path: {test_path}")
    df_test = pd.read_csv(test_path)
    if args.dataset == 'kuairandpure':
        features = [f for f in df_test.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
        label = 'is_click'
    else:
        raise ValueError(args.dataset)
    unique_values = [df_test[col].max()+1 for col in features]
    test_x = df_test[features]
    test_y = df_test[label]
    test_idx = df_test.index
    test_idx = torch.tensor(test_idx.values, dtype=torch.int64)
    test_x, test_y = map(lambda x: torch.tensor(x.values, dtype=torch.long), (test_x, test_y))
    dataset_test = MyDataset(test_x, test_y, test_idx)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers)
    return features, label, test_dataloader, unique_values


def read_from_args_bak01(args, indices_to_drop=set()):
    logging.info(f"train_path: {args.data_path_train}")
    logging.info(f"val_path: {args.data_path_val}")
    logging.info(f"test_path: {args.data_path_test}")
    logging.info(f"full_path: {args.data_path_full}")
    df_train, df_val, df_test, df_full = map(lambda path: pd.read_csv(path), (args.data_path_train, args.data_path_val, args.data_path_test, args.data_path_full))
    
    if args.mode == 'ncv' and indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)
    
    if args.dataset == 'kuairandpure':
        features = [f for f in df_full.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
        label = 'is_click'
    else:
        raise ValueError(args.dataset)
        
    return read_curfull3(args, df_train, df_val, df_test, df_full, features, label)


        
def read_from_args(args, indices_to_drop=set()):
    if args.training_stage in ['1', '2']:
        if args.training_stage == '1':
            args.data_path_train = args.data_path_train_stage_1
            args.data_path_val = args.data_path_val_stage_1
            args.data_path_test = args.data_path_test_stage_1
            args.read_dataset_mode = 'std_train_val_test'
            args.batch_size = args.batch_size_stage_1 if args.batch_size_stage_1 is not None else 4096
            
        elif args.training_stage == '2':
            args.batch_size = args.batch_size_stage_2 if args.batch_size_stage_2 is not None else 4096
            ''' data retrieval '''
            if args.merge_mode_stage_2 == 'single':
                args.read_dataset_mode = 'merge_v02'
                args.data_path_train_t = args.data_path_train_t_stage_2
                args.data_path_train_curday = args.data_path_train_curday_stage_2
                args.data_path_val = args.data_path_val_stage_2
                args.data_path_test = args.data_path_test_stage_2
            elif args.merge_mode_stage_2 == 'multi_day':
                args.read_dataset_mode = 'merge_v03'
                args.retrieval_days = args.retrieval_days_stage_2
                args.data_path_train_curday = args.data_path_train_curday_stage_2
                args.data_path_val = args.data_path_val_stage_2
                args.data_path_test = args.data_path_test_stage_2
            elif args.merge_mode_stage_2 == 'multi_hour':
                args.read_dataset_mode = 'merge_hours'
                args.retrieval_hours = args.retrieval_hours_stage_2
                args.data_path_train_curday = args.data_path_train_curday_stage_2
                args.data_path_val = args.data_path_val_stage_2
                args.data_path_test = args.data_path_test_stage_2
            elif args.merge_mode_stage_2 == 'only_curday':
                args.read_dataset_mode = 'merge_v04'
                args.data_path_train_curday = args.data_path_train_curday_stage_2
                args.data_path_val = args.data_path_val_stage_2
                args.data_path_test = args.data_path_test_stage_2
            else:
                raise ValueError(args.merge_mode_stage_2)

            
    if args.read_dataset_mode == 'merge':
        logging.info(f"train_path_tplus1: {args.data_path_train_1}") 
        logging.info(f"train_path_curday: {args.data_path_train_2}") 
        logging.info(f"val_path: {args.data_path_val}")
        logging.info(f"test_path: {args.data_path_test}")
        df_train_1, df_train_2, df_val, df_test = map(lambda path: pd.read_csv(path), (args.data_path_train_1, args.data_path_train_2, args.data_path_val, args.data_path_test))
        df_train = pd.concat([df_train_1, df_train_2])
    elif args.read_dataset_mode == 'merge_v02':
        logging.info(f"data_train_t: {args.data_path_train_t}") 
        logging.info(f"data_train_curday: {args.data_path_train_curday}")  
        logging.info(f"data_val: {args.data_path_val}")
        logging.info(f"data_test: {args.data_path_test}")
        df_train_t, df_train_cd, df_val, df_test = map(lambda path: pd.read_csv(path), (args.data_path_train_t, args.data_path_train_curday, args.data_path_val, args.data_path_test))
        df_train = pd.concat([df_train_t, df_train_cd])
    elif args.read_dataset_mode == 'merge_v03':
        ''' args.retrieval_days 传入多个检索日期，将所有检索日期合并'''
        logging.info(f"retrieval_days: {args.retrieval_days}")  
        logging.info(f"data_train_curday: {args.data_path_train_curday}")  
        logging.info(f"data_val: {args.data_path_val}")
        logging.info(f"data_test: {args.data_path_test}")
        df_train_cd, df_val, df_test = map(lambda path: pd.read_csv(path), (args.data_path_train_curday, args.data_path_val, args.data_path_test))
        df_train_days_list = [pd.read_csv(os.path.join(args.data_root_pd, str(day), "full.csv")) for day in args.retrieval_days]
        df_train_days_list.append(df_train_cd)
        df_train = pd.concat(df_train_days_list)
    elif args.read_dataset_mode == 'merge_v04':
        logging.info(f"data_train_curday: {args.data_path_train_curday}")  
        logging.info(f"data_val: {args.data_path_val}")
        logging.info(f"data_test: {args.data_path_test}")
        df_train, df_val, df_test = map(lambda path: pd.read_csv(path), (args.data_path_train_curday, args.data_path_val, args.data_path_test))
    elif args.read_dataset_mode == 'merge_hours':
        logging.info(f"retrieval_hours: {args.retrieval_hours}")
        logging.info(f"data_train_curday: {args.data_path_train_curday}")  
        logging.info(f"data_val: {args.data_path_val}")
        logging.info(f"data_test: {args.data_path_test}")
        df_train_cd, df_val, df_test = map(lambda path: pd.read_csv(path), (args.data_path_train_curday, args.data_path_val, args.data_path_test))
        df_train_hours_list = [pd.read_csv(os.path.join(args.data_root_ph, str(hour), "full.csv")) for hour in args.retrieval_hours]
        df_train_hours_list.append(df_train_cd)
        df_train = pd.concat(df_train_hours_list)
    elif args.read_dataset_mode == 'merge_hours_fulltrain':
        logging.info(f"retrieval_hours: {args.retrieval_hours}")
        logging.info(f"data_train_curday: {args.data_path_train_curday}")  
        logging.info(f"full_train_path: {args.data_path_train}")
        logging.info(f"data_val: {args.data_path_val}")
        logging.info(f"data_test: {args.data_path_test}")
        df_full_train, df_train_cd, df_val, df_test = map(lambda path: pd.read_csv(path), (args.data_path_train, args.data_path_train_curday, args.data_path_val, args.data_path_test))
        df_train_hours_list = [pd.read_csv(os.path.join(args.data_root_ph, str(hour), "full.csv")) for hour in args.retrieval_hours]
        df_train_hours_list.extend([df_full_train, df_train_cd])
        df_train = pd.concat(df_train_hours_list)
    elif args.read_dataset_mode == 'std_train_val_test':
        logging.info(f"train_path: {args.data_path_train}")
        logging.info(f"val_path: {args.data_path_val}")
        logging.info(f"test_path: {args.data_path_test}")
        df_train, df_val, df_test = map(lambda path: pd.read_csv(path), (args.data_path_train, args.data_path_val, args.data_path_test))
    elif args.read_dataset_mode == 'std_train_srg_full':
        logging.info(f"train_val_test: {args.data_full}")
        df_full = pd.read_csv(args.data_full)
        df_train, df_val = train_test_split(df_full, test_size=0.1, random_state=42)
        df_test = df_val
    elif args.read_dataset_mode == 'std_train_val_days_test':
        logging.info(f"train_val_days: {args.train_val_days}")
        logging.info(f"test_day: {args.test_day}")
        df_test = pd.read_csv(os.path.join(args.data_root_pd, str(args.test_day), 'full.csv'))
        dfs = [pd.read_csv(os.path.join(args.data_root_pd, str(day), 'full.csv')) for day in args.train_val_days]
        dfs = pd.concat(dfs)
        df_train, df_val = train_test_split(dfs, test_size=0.1, random_state=42)
    elif args.read_dataset_mode == 'std_merge_day':
        logging.info(f"srg_day: {args.srg_day}")
        logging.info(f"rt_day: {args.rt_day}")
        logging.info(f"test_day: {args.test_day}")
        df_test = pd.read_csv(os.path.join(args.data_root_pd, str(args.test_day), 'full.csv'))
        df_srg = pd.read_csv(os.path.join(args.data_root_pd, str(args.srg_day), 'full.csv'))
        df_rt = pd.read_csv(os.path.join(args.data_root_pd, str(args.rt_day), 'full.csv'))
        df = pd.concat([df_srg, df_rt])
        df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    elif args.read_dataset_mode == 'std_train_val_day_test':
        logging.info(f"train_val_days: {args.train_val_day}")
        logging.info(f"test_day: {args.test_day}")
        df_test = pd.read_csv(os.path.join(args.data_root_pd, str(args.test_day), 'full.csv'))
        df = pd.read_csv(os.path.join(args.data_root_pd, str(args.train_val_day), 'full.csv'))
        df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    elif args.read_dataset_mode == 'std_test':
        logging.info(f"test_path: {args.data_path_test}")
        df_test = pd.read_csv(args.data_path_test)
    elif args.read_dataset_mode == 'only_full':
        logging.info(f"full_path: {args.data_path_full}")
        df_full = pd.read_csv(args.data_path_full)
    elif args.read_dataset_mode == 'avazu_hours':
        logging.info(f"data_path_hours_avazu: {args.data_path_hours_avazu}")
        logging.info(f"data_path_train_2: {args.data_path_train_2}")
        train_list = [pd.read_csv(os.path.join(args.data_root_ph, f"{hour}.csv")) for hour in args.data_path_hours_avazu]
        train_list.append(pd.read_csv(args.data_path_train_2))
        df_train = pd.concat(train_list)
        df_val = pd.read_csv(args.data_path_val)
        df_test = pd.read_csv(args.data_path_test)
    else:
        raise ValueError(args.read_dataset_mode)
    
    if args.mode == 'ncv' and indices_to_drop:
        df_train.drop(index=list(indices_to_drop), inplace=True)
    
    if args.dataset == 'kuairandpure':
        label = 'is_click'
    elif args.dataset == 'criteo_conversion_search':
        label = 'Sale'
    elif args.dataset == 'avazu':
        label = 'click'
    else:
        raise ValueError(args.dataset)
    
    with open(os.path.join(args.data_root, 'features_full.json')) as f:
        features = json.load(f)
    with open(os.path.join(args.data_root, 'unique_values_full.json')) as f:
        unique_values = json.load(f)
    unique_values = [int(i) for i in unique_values]
    
    if args.read_dataset_mode == 'only_full':
        return features, label, unique_values
    elif args.read_dataset_mode == 'std_test':
        test_dataloader = get_dataloader_from_df(args, df_test, features, label)
        return features, label, test_dataloader, unique_values
    else:
        train_dataloader, val_dataloader, test_dataloader = map(lambda df: get_dataloader_from_df(args, df, features, label), (df_train, df_val, df_test))
        return features, label, train_dataloader, val_dataloader, test_dataloader, unique_values


def read_raw_data(args):
    logging.info(f"data_day_full_path: {args.data_day_full}")
    df_day_full = pd.read_csv(args.data_day_full)
    
    if args.dataset == 'kuairandpure':
        features = [f for f in df_day_full.columns if f not in ['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms','profile_stay_time','comment_stay_time','is_profile_enter']]
        label = 'is_click'
    else:
        raise ValueError(args.dataset)
    
    x = torch.tensor(df_day_full[features].values, dtype=torch.long)
    y = torch.tensor(df_day_full[label].values, dtype=torch.long)
    idx = torch.tensor(df_day_full.index.values, dtype=torch.long)
    
    return x, y, idx