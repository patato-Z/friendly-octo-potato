import numpy as np
import torch
import torch.nn as nn
import os
import utils.utils as utils
from utils.fs_trainer import modeltrainer
from utils.datasets import read_one_day_full_v04 as read_one_day_full
from utils.datasets import read_one_day, read_from_args, read_raw_data
from models.basemodel import BaseModel
import logging
import json
import datetime

from utils.args import parse_args
from utils.data_retrieval import get_score_sim_model, get_score_sim_dataset_emb


def train_one_day_sw(args, subset_root):
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_one_day_full(args, subset_root)
    model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
    trainer.fit(train_dataloader, val_dataloader)
    trainer.test(test_dataloader, ['auc', 'logloss'])

def train_each_subset_sw(args):

    ''' train each day offline '''
    assert args.info_str is not None
    subset_root_list = [os.path.join(args.data_root_sw, d) for d in os.listdir(args.data_root_sw)]
    for subset_root in subset_root_list:
        args.save_path = subset_root
        train_one_day_sw(args, subset_root)

def test_delay(args):
    offline_idx = args.offline_idx if args.offline_idx is not None else 2
    max_delay = args.max_delay if args.max_delay is not None else 2
    model_name = args.info_str if args.info_str is not None else 'model_offline'
    # args.read_test_only = True
    
    if args.dataset == 'kuairandpure':
        max_idx = 10
    else:
        raise ValueError(args.dataset)
    assert offline_idx + max_delay <= max_idx
    
    model_path = os.path.join(args.data_root_sw, str(offline_idx), f'{model_name}.pth')
    logging.info(f"model path: {model_path}")
    # test_root_list = [os.path.join(args.data_root_sw, str(offline_idx + i)) for i in range(max_delay)]  # v01
    test_root_list = [os.path.join(args.data_root_sw, str(offline_idx + i)) for i in range(1, max_delay + 1)]  # v02
    for subset_root in test_root_list:
        logging.info(f"subset_root: {subset_root}")
        features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_one_day_full(args, subset_root)
        # features, label, test_dataloader, unique_values = read_one_day(args, subset_root)
        model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
        model.load_state_dict(torch.load(model_path))
        trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
        trainer.test(test_dataloader, ['auc', 'logloss'])
    

def data_retrieval(args):
    
    subset_root = os.path.join(args.data_root_sw, str(args.subset_idx))
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_one_day_full(args, subset_root)
    model_srg = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    trainer = modeltrainer(args, model_srg, args.model, args.device, retrain=False)
    with open(os.path.join(subset_root, 'info.txt')) as f:
        lines = f.readlines()
        for line in lines:
            if "next_date" in line:
                next_date_str = line.split("=")[1].strip()
            elif "start_date" in line:
                start_date_str = line.split("=")[1].strip()
            elif "end_date" in line:
                end_date_str = line.split("=")[1].strip()
    logging.info(f"next_date = {next_date_str}")
    logging.info(f"start_date = {start_date_str}")
    logging.info(f"end_date = {end_date_str}")
    srg_path = os.path.join(args.data_root_pd, next_date_str, f"{args.load_srg}.pth")
    model_srg.load_state_dict(torch.load(srg_path))
    days_full_list = sorted(os.listdir(args.data_root_pd))
    days_subset_list = [date for date in days_full_list if start_date_str <= date <= end_date_str]
    logging.info("start getting score with srg")
    args.test_file_name = 'full'
    args.read_mode = 'test_only'
    score_ori_list = []
    for day in days_subset_list:
        srg_path_day = os.path.join(args.data_root_pd, day, f"{args.load_srg}.pth")
        model_srg_day = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
        model_srg_day.to(torch.device(args.device))
        model_srg_day.load_state_dict(torch.load(srg_path_day))
        score = get_score_sim_model(model_srg, model_srg_day)
        score_ori_list.append(score)
        logging.info(f"srg path: {srg_path_day}")
        logging.info(f"score: {score:.5f}")
        
    _, indices = torch.topk(torch.tensor(score_ori_list), k=5, largest=False)
    topk_days = torch.tensor([int(i) for i in days_subset_list])[indices]
    for topk_day in topk_days:
        topk_day_str = str(topk_day.item())
        
    assert len(days_subset_list) == len(score_ori_list)
    score_list = []
    for auc, logloss in score_ori_list:
        pass
    
    model_ft_path = os.path.join(subset_root, f"{args.load_offline}.pth")
    logging.info(f"model load for fine-tune: {model_ft_path}")
    args.train_mode = 'finetune'
    args.read_dataset_mode = 'single'
    score_list_ft = []
    logging.info("start baseline")
    args.data_path_train = os.path.join(args.data_root_pd, next_date_str, 'full.csv')  # t+1天的数据
    args.data_path_val = os.path.join(args.data_root_pd, next_date_str, 'full.csv')
    args.data_path_test = os.path.join(args.data_root_pd, next_date_str, 'full.csv')
    args.data_path_full = 'data/kuairand-pure/preprocessed-kuairandpure.csv'
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
    model_ft = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    model_ft.load_state_dict(torch.load(model_ft_path))
    trainer = modeltrainer(args, model_ft, args.model, args.device, retrain=False)
    trainer.fit(train_dataloader, val_dataloader)
    auc_ft, logloss_ft = trainer.test(test_dataloader, ['auc', 'logloss'])
    logging.info(f"day t+1: {next_date_str}")
    logging.info(f"auc t+1: {auc_ft}")
    logging.info(f"logloss t+1: {logloss_ft}")    
    
    logging.info("start fine-tune")
    args.read_dataset_mode = 'merge'
    for day in days_subset_list:
        args.data_path_train_tplus1 = os.path.join(args.data_root_pd, next_date_str, 'full.csv')  # t+1天的数据
        args.data_path_train_curday = os.path.join(args.data_root_pd, day, 'full.csv')  # 当天的数据
        args.data_path_val = os.path.join(args.data_root_pd, next_date_str, 'full.csv')
        args.data_path_test = os.path.join(args.data_root_pd, next_date_str, 'full.csv')
        args.data_path_full = 'data/kuairand-pure/preprocessed-kuairandpure.csv'
        features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
        model_ft = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
        model_ft.load_state_dict(torch.load(model_ft_path))
        trainer = modeltrainer(args, model_ft, args.model, args.device, retrain=False)
        trainer.fit(train_dataloader, val_dataloader)
        auc_ft, logloss_ft = trainer.test(test_dataloader, ['auc', 'logloss'])
        logging.info(f"day: {day}")
        logging.info(f"auc_ft: {auc_ft}")
        logging.info(f"logloss_ft: {logloss_ft}")
        score_list_ft.append((auc_ft, logloss_ft))
    ###### tmp end
    print()


def train_per_day_srg(args):
    assert args.info_str is not None
    subset_root_list = [os.path.join(args.data_root_pd, d) for d in os.listdir(args.data_root_pd)]
    for subset_root in subset_root_list:
        args.save_path = os.path.join(subset_root, f"{args.info_str}.pth")
        features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_one_day_full(args, subset_root)
        model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
        trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
        trainer.fit(train_dataloader, val_dataloader)


def observation_data_retrieval(args):
    subset_root = os.path.join(args.data_root_sw, str(args.subset_idx))
    args.train_mode = 'finetune'
    args.read_dataset_mode = 'single'
    logging.info("start baseline")
    with open(os.path.join(subset_root, 'info.txt')) as f:
        lines = f.readlines()
        for line in lines:
            if "next_date" in line:
                next_date_str = line.split("=")[1].strip()
            elif "start_date" in line:
                start_date_str = line.split("=")[1].strip()
            elif "end_date" in line:
                end_date_str = line.split("=")[1].strip()
    logging.info(f"next_date = {next_date_str}")
    logging.info(f"start_date = {start_date_str}")
    logging.info(f"end_date = {end_date_str}")
    
    base_model = os.path.join(args.data_root_pd, end_date_str, f"{args.load_srg}.pth")
    days_full_list = sorted(os.listdir(args.data_root_pd))
    days_subset_list = [date for date in days_full_list if start_date_str <= date < end_date_str]
    
    logging.info("start fine-tune")
    args.read_dataset_mode = 'merge'
    auc_ft_list = []
    logloss_ft_list = []
    for day in days_subset_list:
        args.data_path_train_tplus1 = os.path.join(args.data_root_pd, end_date_str, 'full.csv')  # t天的数据
        args.data_path_train_curday = os.path.join(args.data_root_pd, day, 'full.csv')  # 当天的数据
        args.data_path_val = os.path.join(args.data_root_pd, next_date_str, 'full.csv')
        args.data_path_test = os.path.join(args.data_root_pd, next_date_str, 'full.csv')
        args.data_path_full = 'data/kuairand-pure/preprocessed-kuairandpure.csv'
        
        args.save_path = os.path.join(args.log_path, f"ft-base_{end_date_str}-rt_{day}.pth")
        # 这里要融合两个子集的数据
        features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
        model_ft = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
        model_ft.load_state_dict(torch.load(base_model))
        trainer = modeltrainer(args, model_ft, args.model, args.device, retrain=False)
        trainer.fit(train_dataloader, val_dataloader)
        auc_ft, logloss_ft = trainer.test(test_dataloader, ['auc', 'logloss'])
        logging.info(f"day: {day}")
        logging.info(f"auc_ft: {auc_ft}")
        logging.info(f"logloss_ft: {logloss_ft}")
        auc_ft_list.append(auc_ft)
        logloss_ft_list.append(logloss_ft)
    
    args.read_dataset_mode = 'single'
    args.data_path_train = os.path.join(args.data_root_pd, end_date_str, 'full.csv')  # t+1天的数据
    args.data_path_val = os.path.join(args.data_root_pd, end_date_str, 'full.csv')
    args.data_path_test = os.path.join(args.data_root_pd, next_date_str, 'full.csv')
    args.data_path_full = 'data/kuairand-pure/preprocessed-kuairandpure.csv'
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
    model_base = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    model_base.load_state_dict(torch.load(base_model))
    trainer = modeltrainer(args, model_ft, args.model, args.device, retrain=False)
    trainer.fit(train_dataloader, val_dataloader)
    auc_base, logloss_base = trainer.test(test_dataloader, ['auc', 'logloss'])
    logging.info(f"day t: {end_date_str}")
    logging.info(f"auc base: {auc_base}")
    logging.info(f"logloss base: {logloss_base}")    
    
    print()    


def sim_m2m(args):
    get_data_root(args)
    args.read_dataset_mode = 'single'

    end_date_str = "20220507"
    
    base_model = os.path.join(args.data_root_pd, end_date_str, f"{args.load_srg}.pth")
    days_full_list = sorted(os.listdir(args.data_root_pd))
    days_subset_list = [date for date in days_full_list if date < end_date_str]
    
    args.read_dataset_mode = 'only_full'
    features, label, unique_values = read_from_args(args)
    model_base = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    model_base.load_state_dict(torch.load(base_model))
    model_base.to(torch.device(args.device))
    
    sim_score = []
    for day in days_subset_list:
        model_rt = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
        model_rt_path = os.path.join(args.data_root_pd, day, f"{args.load_srg}.pth")
        model_rt.load_state_dict(torch.load(model_rt_path))
        model_rt.to(torch.device(args.device))
        score = get_score_sim_model(model_base, model_rt)
        logging.info(f"day: {day}, score: {score}")
        sim_score.append(score)
    
    print()


def sim_m2d(args):
    get_data_root(args)
    args.srg_model_date = "20220507"
    args.log_file_name = f"srg_date_{args.srg_model_date}-srg_name_{args.load_srg}"
    set_log(args)
    
    args.read_dataset_mode = 'only_full'
    features, label, unique_values = read_from_args(args)
    model_srg = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    srg_model = os.path.join(args.data_root_pd, args.srg_model_date, f"{args.load_srg}.pth")
    model_srg.load_state_dict(torch.load(srg_model))
    model_srg.to(torch.device(args.device))
    
    
    results = []
    args.read_dataset_mode = 'single'
    days_full_list = sorted(os.listdir(args.data_root_pd))
    end_date_str = "20220507"
    days_subset_list = [date for date in days_full_list if date < end_date_str]
    with torch.no_grad():
        for day in days_subset_list:
            day_full_path = os.path.join(args.data_root_pd, day, 'full.csv')
            args.data_path_train = day_full_path
            args.data_path_val = day_full_path
            args.data_path_test = day_full_path

            features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
            trainer = modeltrainer(args, model_srg, args.model, args.device, retrain=False)
            auc, logloss = trainer.test(test_dataloader, ['auc', 'logloss'])
            results.append((args.srg_model_date, day, auc, logloss))
    logging.info(results)
    print()


def sim_d2d_emb(args):
    get_data_root(args)
    set_log(args)

    end_date_str = "20220507"
    args.data_day_full = os.path.join(args.data_root_pd, end_date_str, 'full.csv')
    x_base, y_base, idx_base = read_raw_data(args)
    
    args.read_dataset_mode = 'only_full'
    features, label, unique_values = read_from_args(args)
    model_srg = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    srg_model = os.path.join(args.data_root_pd, end_date_str, f"{args.load_srg}.pth")
    model_srg.load_state_dict(torch.load(srg_model))
    # model_srg.to(torch.device(args.device))
    
    days_full_list = sorted(os.listdir(args.data_root_pd))
    days_subset_list = [date for date in days_full_list if date < end_date_str]
    score_list = []
    for day in days_subset_list:
        args.data_day_full = os.path.join(args.data_root_pd, day, 'full.csv')
        x_day, y_day, idx_day = read_raw_data(args)
        score_x = get_score_sim_dataset_emb(x_base, x_day, model_srg)
        logging.info(score_x)
        score_list.append(score_x)
    logging.info(score_list)
    print()


def train_baseline(args):
    get_data_root(args)
    
    args.log_file_name = f"train_baseline-step-epoch_{args.early_stop_epoch}-step_{args.early_stop_step_step}-ps_{args.patience_step}"
    
    set_log(args)

    args.read_dataset_mode = 'single'
    data_root = 'data/kuairand-pure/idea-v02/01'
    args.data_path_train = os.path.join(data_root, 'train.csv')
    args.data_path_val = os.path.join(data_root, 'val.csv')
    args.data_path_test = os.path.join(data_root, 'test.csv')
    model_save_dir = os.path.join(data_root, 'ckpt')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    args.save_path = os.path.join(model_save_dir, f"baseline-e_{args.epoch}.pth")
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
    model_baseline = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    trainer = modeltrainer(args, model_baseline, args.model, args.device, retrain=False)
    trainer.fit(train_dataloader, val_dataloader)
    trainer.test(test_dataloader, ['auc', 'logloss'])


def finetune_gt(args):
    set_log(args)
    end_date_str = "20220507"
    
    days_full_list = sorted(os.listdir(args.data_root_pd))
    days_subset_list = [date for date in days_full_list if date < end_date_str]
    
    args.read_dataset_mode = 'only_full'
    args.data_path_full = 'data/kuairand-pure/preprocessed-kuairandpure.csv'
    features, label, unique_values = read_from_args(args)
    
    load_ckpt_dir = "data/kuairand-pure/idea-v02/01/ckpt"
    base_model_dict = {}
    for i in [1, 2, 3]:
        ckpt_path = os.path.join(load_ckpt_dir, f"baseline-e_{i}.pth")
        model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
        model.to(torch.device(args.device))
        model.load_state_dict(torch.load(ckpt_path))
        base_model_dict[i] = model
    
    args.read_dataset_mode = 'merge_v02'
    args.data_path_train_t = os.path.join(args.data_root_pd, end_date_str, "full.csv")
    data_root = 'data/kuairand-pure/idea-v02/01'
    args.data_path_val = os.path.join(data_root, 'val.csv')
    args.data_path_test = os.path.join(data_root, 'test.csv')
    result = []
    for i, model in base_model_dict.items():
        for day in days_subset_list:
            logging.info(f"baseline-e_{i}")
            logging.info(f"day: {day}")
            args.data_path_train_curday = os.path.join(args.data_root_pd, day, "full.csv")
            _, _, train_dataloader, val_dataloader, test_dataloader, _ = read_from_args(args)
            trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
            trainer.fit(train_dataloader, val_dataloader)
            auc, logloss = trainer.test(test_dataloader, ['auc', 'logloss'])
            result.append((i, day, auc, logloss))
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(f"ft-data-{timestamp}.json", "w") as file:
        json.dump(result, file)
    print()
        

def get_data_root(args):
    if args.dataset == 'kuairandpure':
        args.data_root = 'data/kuairand-pure/idea-v02/01'
        args.data_root_sw = f'data/kuairand-pure/slide_window/width_{args.sw_width}'
        args.data_root_pd = 'data/kuairand-pure/per_day'
        args.data_path_full = 'data/kuairand-pure/preprocessed-kuairandpure.csv'
    elif args.dataset == 'criteo_conversion_search':
        args.data_root = 'data/Criteo_Conversion_Search'
        args.data_root_pd = 'data/Criteo_Conversion_Search/per_day'
        args.data_root_ph = 'data/Criteo_Conversion_Search/per_hour/v02'
    elif args.dataset == 'avazu':
        args.data_root = 'data/avazu'
        args.data_root_pd = 'data/avazu/per_day'
        args.data_root_ph = 'data/avazu/per_hour'
    else:
        raise ValueError(args.dataset)    
    
def get_args_data_path_finetune(args):
    get_data_root(args)
    
    data_root = 'data/kuairand-pure/idea-v02/01'
    args.data_path_train = os.path.join(data_root, 'train.csv')
    args.data_path_val = os.path.join(data_root, 'val.csv')
    args.data_path_test = os.path.join(data_root, 'test.csv')
    args.data_path_full = 'data/kuairand-pure/preprocessed-kuairandpure.csv'
    
    end_date_str = "20220507"
    args.data_path_train_curday = os.path.join(args.data_root_pd, end_date_str, "full.csv")
    args.load_ckpt_dir = "data/kuairand-pure/idea-v02/01/ckpt"
    
    
def finetune(args):
    get_args_data_path_finetune(args)
    
    # args.retrieval_days : list
    args.retrieval_days = eval(args.retrieval_days_rank)[:args.retrieval_days_num]
    load_baseline = f"baseline-e_{args.load_baseline}"
    logging.info(load_baseline)
    
    args.read_dataset_mode = 'merge_v03'
    features, _, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
    model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    load_ckpt = os.path.join(args.load_ckpt_dir, f"{load_baseline}.pth")
    logging.info(f"load ckpt: {load_ckpt}")
    model.load_state_dict(torch.load(load_ckpt))
    trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
    trainer.fit(train_dataloader, val_dataloader)
    trainer.test(test_dataloader, ['auc', 'logloss'])



def std_train_val_test(args):
    get_data_root(args)
    set_log(args)
    args.read_dataset_mode = 'std_train_val_test' if args.read_dataset_mode is None else args.read_dataset_mode
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
    model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
    if args.std_load_model is not None:
        logging.info(f"load model: {args.std_load_model}")
        model.load_state_dict(torch.load(args.std_load_model))
    trainer.fit(train_dataloader, val_dataloader)
    trainer.test(test_dataloader, ['auc', 'logloss'])


def std_test(args):
    get_data_root(args)
    set_log(args)
    args.read_dataset_mode = 'std_test' if args.read_dataset_mode is None else args.read_dataset_mode
    features, label, test_dataloader, unique_values = read_from_args(args)
    model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    logging.info(f"load model: {args.std_load_model}")
    model.load_state_dict(torch.load(args.std_load_model))
    trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
    trainer.test(test_dataloader, ['auc', 'logloss']) 


def train_two_stage(args):
    get_data_root(args)
    set_log(args)
    
    
    args.training_stage = '1'
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
    model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
    trainer.fit(train_dataloader, val_dataloader)

    args.training_stage = '2'
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_from_args(args)
    trainer = modeltrainer(args, model, args.model, args.device, retrain=False)
    trainer.fit(train_dataloader, val_dataloader)
    trainer.test(test_dataloader, ['auc', 'logloss'])

    

def set_log(args):
    assert args.exp_id is not None
    log_path = os.path.join(args.log_path, args.exp_id) if args.log_path is not None else f'exp/{args.exp_id}/'
    args.log_path = log_path
    os.makedirs(log_path, exist_ok=True)
    
    if args.log_file_name_mode == 'debug-timestamp':
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        args.log_file_name = f'debug-{timestamp}'
    elif args.log_file_name_mode == 'args':
        assert args.log_file_name
    elif args.log_file_name_mode == 'std_step':
        args.log_file_name = f'{args.early_stop_mode}_{args.early_stop_epoch}_{args.early_stop_step_step}_{args.patience_step}'
    else:
        raise ValueError(args.log_file_name_mode)
    
    log_file_name = args.log_file_name if args.log_file_name is not None else 'log'
    if args.log_use_timestamp:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        log_file_name = f'{log_file_name}--{timestamp}.log'
    else:
        log_file_name = f'{log_file_name}.log'
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(os.path.join(log_path, log_file_name), mode='w'),
                            logging.StreamHandler(),
                        ])
    logging.info(args)
    logging.info(f"mode: {args.mode}")
    


def main(args):

    if args.seed != 0:
        utils.seed_everything(args.seed)
    
    assert args.mode is not None
    if args.mode == 'train_each_subset_sw':
        set_log(args)
        train_each_subset_sw(args)
        return
    elif args.mode == 'train_one_day_sw':
        set_log(args)
        subset_root = args.save_path
        logging.info(f"subset // save root : {subset_root}")
        train_one_day_sw(args, subset_root)
        return
    elif args.mode == 'test_delay':
        set_log(args)
        test_delay(args)
        return
    elif args.mode == 'data_retrieval':
        set_log(args)
        data_retrieval(args)
        return
    elif args.mode == 'train_per_day_srg':
        set_log(args)
        train_per_day_srg(args)
        return
    elif args.mode == 'observation_data_retrieval':
        set_log(args)
        observation_data_retrieval(args)
        return
    elif args.mode == 'sim_m2m':
        set_log(args)
        sim_m2m(args)
        return
    elif args.mode == 'sim_m2d':
        sim_m2d(args)
        return
    elif args.mode == 'sim_d2d_emb':
        sim_d2d_emb(args)
        return
    elif args.mode == 'train_baseline':
        train_baseline(args)
        return
    elif args.mode == 'finetune_gt':
        finetune_gt(args)
        return
    elif args.mode == 'finetune':
        # args.log_use_timestamp = True'
        if args.early_stop_mode == 'epoch':
            args.log_file_name = f"{args.retrieval_method}-topk_{args.retrieval_days_num}-fte_{args.epoch}-b_{args.load_baseline}"
        elif args.early_stop_mode == 'step':
            args.log_file_name = f"{args.retrieval_method}-topk_{args.retrieval_days_num}-fts_e_{args.early_stop_epoch}-step_{args.early_stop_step_step}-p_{args.patience_step}-b_{args.load_baseline}"
        else:
            raise ValueError(args.early_stop_mode)
        set_log(args)
        finetune(args)
        return
    elif args.mode == 'std_train_val_test':
        std_train_val_test(args)
    elif args.mode == 'std_test':
        std_test(args)
    elif args.mode == 'train_two_stage':
        train_two_stage(args)        
    else:
        raise ValueError(args.mode)
    
    logging.info(f"exp id: {args.exp_id}")
    logging.info(f"log_file_name: {args.log_file_name}")


if __name__ == '__main__':

    args = parse_args()
    main(args)