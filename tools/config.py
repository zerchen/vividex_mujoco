import os
import os.path as osp
import sys
from yacs.config import CfgNode as CN
from loguru import logger
from contextlib import redirect_stdout

vid_mapping = {
        "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211": "mb1", 
        "ycb-006_mustard_bottle-20200908-subject-05-20200908_144439": "mb2", 
        "ycb-006_mustard_bottle-20200928-subject-07-20200928_144226": "mb3", 
        "ycb-005_tomato_soup_can-20200709-subject-01-20200709_142853": "tc1", 
        "ycb-005_tomato_soup_can-20201015-subject-09-20201015_143403": "tc2", 
        "ycb-005_tomato_soup_can-20200709-subject-01-20200709_142926": "tc3", 
        "ycb-004_sugar_box-20200918-subject-06-20200918_113441": "sb1", 
        "ycb-004_sugar_box-20200903-subject-04-20200903_104157": "sb2", 
        "ycb-004_sugar_box-20200709-subject-01-20200709_142553": "sb3", 
        "ycb-052_extra_large_clamp-20200709-subject-01-20200709_152843": "ec1", 
        "ycb-052_extra_large_clamp-20200820-subject-03-20200820_144829": "ec2", 
        "ycb-052_extra_large_clamp-20201002-subject-08-20201002_112816": "ec3", 
        "ycb-025_mug-20200709-subject-01-20200709_150949": "mu1", 
        "ycb-025_mug-20200928-subject-07-20200928_154547": "mu2", 
        "ycb-025_mug-20200820-subject-03-20200820_143304": "mu3", 
    }

extra_vid_mapping = {
        "ycb-005_tomato_soup_can-20201002-subject-08-20201002_105343": "tc4", 
    }

cfg = CN()

cfg.task = 'visual_policy'
cfg.video = 'ycb-006_mustard_bottle-20200709-subject-01-20200709_143211'
cfg.test_video = 'ycb-006_mustard_bottle-20200709-subject-01-20200709_143211'
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '..')
cfg.data_dir = osp.join(cfg.root_dir, 'datasets')
cfg.output_dir = '.'
cfg.model_dir = './model_dump'
cfg.vis_dir = './vis'
cfg.log_dir = './log'
cfg.result_dir = './result'

## dataset
cfg.trainset = 'dexycb_traj'

## model setting
cfg.point_num = 2048
cfg.obs_dim = 289
cfg.mlp_dim = [256, 256]
cfg.pc_latent_dim = 256
cfg.act_dim = 30
cfg.pointnet_type = 'pointnet'
cfg.point_cs = 'world'
cfg.point_noise = False
cfg.complete_pc = False

## training config
cfg.warm_up_epoch = 0
cfg.lr_dec_epoch = [2000, 3000]
cfg.end_epoch = 2000
cfg.lr = 1e-5
cfg.lr_dec_style = 'step'
cfg.lr_dec_factor = 0.5
cfg.train_batch_size = 16

## others
cfg.num_threads = 5
cfg.gpu_ids = (0, 1, 2, 3)
cfg.num_gpus = 4
cfg.checkpoint = 'model.pth.tar'
cfg.model_save_freq = 100

def update_config(cfg, args, mode='train'):
    cfg.defrost()
    if mode == "test":
        cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.gpu_ids = args.gpu_ids
    cfg.num_gpus = len(cfg.gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info('>>> Using GPU: {}'.format(cfg.gpu_ids))

    if cfg.video in ["pour", "place"]:
        abbr_names = [cfg.video]
    else:
        if '/' not in cfg.video:
            seq_list = list(vid_mapping.keys())
            if cfg.video == "subset":
                seq_chosen = seq_list[0:-1:3]
            elif cfg.video == "allset":
                seq_chosen = seq_list
            else:
                seq_chosen = []
                for seq_name in seq_list:
                    if cfg.video in seq_name:
                        seq_chosen.append(seq_name)
            cfg.video = '/'.join(seq_chosen)

        if '/' not in cfg.test_video:
            seq_list = list(vid_mapping.keys())
            if cfg.test_video == "subset":
                seq_chosen_test = seq_list[0:-1:3]
            elif cfg.test_video == "allset":
                seq_chosen_test = seq_list
            else:
                seq_chosen_test = []
                for seq_name in seq_list:
                    if cfg.test_video in seq_name:
                        seq_chosen_test.append(seq_name)
            cfg.test_video = '/'.join(seq_chosen_test)

        original_video_names = cfg.video.split('/')
        abbr_names = []
        for name in original_video_names:
            try:
                abbr_names.append(vid_mapping[name])
            except:
                abbr_names.append(extra_vid_mapping[name])

    if mode == 'train':
        exp_info = [cfg.trainset, '-'.join(abbr_names), cfg.pointnet_type, cfg.point_cs, 'l' + str(cfg.pc_latent_dim), 'd' + str(cfg.mlp_dim[0]), 'e' + str(cfg.end_epoch), 'b' + str(cfg.num_gpus * cfg.train_batch_size), 'num' + str(cfg.point_num), 'cp' + str(int(cfg.complete_pc)), 'np' + str(int(cfg.point_noise))]

        cfg.output_dir = osp.join(cfg.cur_dir, 'outputs', cfg.task, '_'.join(exp_info))
        cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.log_dir = osp.join(cfg.output_dir, 'log')
        cfg.result_dir = osp.join(cfg.output_dir, 'result')

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.vis_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)

        cfg.freeze()
        with open(osp.join(cfg.output_dir, 'exp.yaml'), 'w') as f:
            with redirect_stdout(f): print(cfg.dump())
    elif mode == "test":
        cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.log_dir = osp.join(cfg.output_dir, 'log')
        cfg.result_dir = osp.join(cfg.output_dir, 'result')

        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.vis_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)

        cfg.freeze()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset))
