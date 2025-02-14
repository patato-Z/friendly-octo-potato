import argparse
import yaml
import time
import nni
from . import utils

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avazu')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--fs', type=str, default='no_selection')
    parser.add_argument('--es', type=str, default='no_selection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device',type=str, default='cuda:0')
    
    # ncv
    parser.add_argument('--ncv_iters', type=int, default=10, help='ncv_iters')
    parser.add_argument('--drop_ratio_iter', type=float, default=0.01)
    parser.add_argument('--min_remain_ratio', type=float, default=0.8)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--epoch_ft', type=int, default=100)
    parser.add_argument('--save_ckpt', type=bool, default=False)
    parser.add_argument('--mode_debug', action='store_true')
    parser.add_argument('--topk_func', type=str, default='loss')
    parser.add_argument('--mode', type=str, default=None)
    # for log file
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--exp_id', type=str, default=None)
    parser.add_argument('--log_file_name', type=str, default=None)
    parser.add_argument('--log_file_name_mode', type=str, default=None)
    parser.add_argument('--skip_args_fields', nargs='+', type=str, help='List of fruit names', default=None)
    
    # for finetune
    parser.add_argument('--retrieval_method', type=str, default=None)
    parser.add_argument('--retrieval_days_rank', type=str, default=None)
    parser.add_argument('--retrieval_days_num', type=int, default=1)
    parser.add_argument('--load_baseline', type=int, default=1)
    # for train 2 stage
    parser.add_argument('--load_config', type=str, default=None)
    
    parser.add_argument('--training_stage', type=str, default=None)
    parser.add_argument('--loss_func', type=str, default='bce')
    
    parser.add_argument('--data_path_train_stage_1', type=str, default=None)
    parser.add_argument('--data_path_val_stage_1', type=str, default=None)
    parser.add_argument('--data_path_test_stage_1', type=str, default=None)
    
    parser.add_argument('--merge_mode_stage_2', type=str, default=None)
    parser.add_argument('--data_path_train_t_stage_2', type=str, default=None)
    parser.add_argument('--data_path_train_curday_stage_2', type=str, default=None)
    parser.add_argument('--data_path_val_stage_2', type=str, default=None)
    parser.add_argument('--data_path_test_stage_2', type=str, default=None)
    
    parser.add_argument('--data_path_hours_avazu', nargs='+')
    
    parser.add_argument('--optim_stage_1', type=str, default=None)
    parser.add_argument('--optim_stage_2', type=str, default=None)
    parser.add_argument('--lambda_l2_stage_1', type=float, default=None)
    parser.add_argument('--lambda_l2_stage_2', type=float, default=None)
    parser.add_argument('--lr_stage_1', type=float, default=None)
    parser.add_argument('--lr_stage_2', type=float, default=None)
    parser.add_argument('--epoch_stage_1', type=int, default=None)
    parser.add_argument('--epoch_stage_2', type=int, default=None)
    
    parser.add_argument('--batch_size_stage_1', type=int, default=None)
    parser.add_argument('--batch_size_stage_2', type=int, default=None)
    
    # for model load std
    parser.add_argument('--std_load_model', type=str, default=None)
    
    parser.add_argument('--train_val_days', type=str, default=None)
    parser.add_argument('--train_val_day', type=str, default=None)
    parser.add_argument('--test_day', type=str, default=None)
    parser.add_argument('--srg_day', type=str, default=None)
    parser.add_argument('--rt_day', type=str, default=None)
    
    # for model save
    parser.add_argument('--save_mode', type=str, default=None)  
    parser.add_argument('--save_path', type=str, default=None)  
    # for dataset read
    parser.add_argument('--read_dataset_mode', type=str, default=None)
    parser.add_argument('--data_path_train', type=str, default=None)
    parser.add_argument('--data_path_val', type=str, default=None)
    parser.add_argument('--data_path_test', type=str, default=None)
    parser.add_argument('--data_full', type=str, default=None)
    parser.add_argument('--data_path_train_1', type=str, default=None)
    parser.add_argument('--data_path_train_2', type=str, default=None)
    
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--data_root_pd', type=str, default=None)
    parser.add_argument('--data_root_ph', type=str, default=None)
    
    
    parser.add_argument('--train_file_name', type=str, default=None)
    parser.add_argument('--val_file_name', type=str, default=None)
    parser.add_argument('--test_file_name', type=str, default=None)
    parser.add_argument('--get_full', type=str, default=None, help='method to get full')
    parser.add_argument('--read_mode', type=str, default=None)
    
    # for test_delay
    parser.add_argument('--max_delay', type=int, default=None)
    parser.add_argument('--offline_idx', type=int, default=None)
    # for data_retrieval
    parser.add_argument('--subset_idx', type=int, default=None)
    parser.add_argument('--load_srg', type=str, default=None)
    parser.add_argument('--load_offline', type=str, default=None)
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--retrieval_days', type=str, default='')
    # for early stop for step
    parser.add_argument('--early_stop_mode', type=str, default='epoch')
    parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
    parser.add_argument('--patience_step', type=int, default=100000, help='early stopping patience for step')
    parser.add_argument('--early_stop_epoch', type=int, default=0)
    parser.add_argument('--early_stop_step_step', type=int, default=10)
    parser.add_argument('--early_stop_start_step', type=int, default=0)
    
    # slide window
    parser.add_argument('--sw_previous_len', type=int, default=20)  # v01
    parser.add_argument('--sw_width', type=int, default=20)  # v02
    parser.add_argument('--sw_t', type=int, default=1)
    parser.add_argument('--sw_read_idx', type=int, default=0)
    

    parser.add_argument('--data_path', type=str, default='data/', help='data path') # ~/autodl-tmp/ or data/
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('--dataset_shuffle', type=bool, default=True, help='whether to shuffle the dataset')
    parser.add_argument('--embedding_dim', type=int, default=8, help='embedding dimension')
    parser.add_argument('--train_or_search', type=utils.str2bool, default=True, help='whether to train or search')
    parser.add_argument('--retrain', type=utils.str2bool, default=True, help='whether to retrain')
    parser.add_argument('--k', type=int, default=0, help='top k features')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_min', type=float, default=1.e-4, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--nni', type=bool, default=False, help='whether to use nni')
    parser.add_argument('--rank_path', type=str, default='None', help='if only retrain, no train, please specify the path of feature_rank file. e.g., autofield_no_selection_avazu')
    parser.add_argument('--read_feature_rank', type=utils.str2bool, default=True, help='whether to use pre-saved feature rank')

    args = parser.parse_args()
    
    with open('models/config.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    args.__dict__.update(model_config)

    if args.load_config is not None:
        with open(args.load_config, 'r') as file:
            config = yaml.safe_load(file)
        if args.skip_args_fields is not None:
            for key, value in config.items():
                if key not in args.skip_args_fields:
                    args.__dict__[key] = value
        else:
            args.__dict__.update(config)
    
    args.timestr = str(time.time())
    args.log_use_timestamp = False
    return args
    
