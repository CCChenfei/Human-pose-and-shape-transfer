import sys
from absl import flags
import os.path as osp
from os import makedirs
from glob import glob
from datetime import datetime
import json

import numpy as np

curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()
SMPL_MODEL_PATH = osp.join(model_dir, 'neutral_smpl_with_cocoplus_reg.pkl')
SMPL_FACE_PATH = osp.join(model_dir, 'smpl_faces.npy')
SMPL_COLOR_PATH=osp.join(model_dir,'vertex.npz')

# Default pred-trained model path for the demo.
PRETRAINED_MODEL = osp.join(model_dir, 'model.ckpt-667589')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neurtral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')
flags.DEFINE_string('smpl_color_path', SMPL_COLOR_PATH,
                    'path to smpl mesh faces (for easy rendering)')
flags.DEFINE_string('load_path', None, 'path to trained model')
flags.DEFINE_string('pretrained_model_path', None,
                    'if not None, fine-tunes from this ckpt')

flags.DEFINE_string('color_pretrained_model_path', None,
                    'if not None, fine-tunes from this ckpt')
flags.DEFINE_string('feature_loss_pretrained_model_path', None,'if not None, fine-tunes from this ckpt')

flags.DEFINE_string('discriminator_type','norm','dis type')
flags.DEFINE_string('discriminator_loss_type','norm','dis loss type')
flags.DEFINE_integer('batch_size', 8,
                     'Input image size to the network after preprocessing')
flags.DEFINE_integer('num_images', 770000,
                     'Input image size to the network after preprocessing')
flags.DEFINE_integer('useCamBatches',1,
                     'how many views to use')

# Don't change if testing:
flags.DEFINE_integer('img_size', 224,
                     'Input image size to the network after preprocessing')
flags.DEFINE_string('data_format', 'NHWC', 'Data format')
flags.DEFINE_integer('num_stage', 3, '# of times to iterate regressor')
flags.DEFINE_integer('num_cam', 3, '# of times to iterate regressor')
flags.DEFINE_integer('num_theta', 72, '# of times to iterate regressor')
flags.DEFINE_integer('joints_num', 14, '# of epochs to train')
flags.DEFINE_integer('color_length', 256, '# of times to iterate regressor')

flags.DEFINE_string(
    'joint_type', 'cocoplus',
    'cocoplus (19 keypoints) or lsp 14 keypoints, returned by SMPL')

# Training settings:
# TODO! If you want to train, change this to your 'tf_datasets' or specify it with the flag.
DATA_DIR = '/home/gzp/Personal/dataset/H36M_RES512/train'

flags.DEFINE_string('data_dir', DATA_DIR, 'Where to save training models')
flags.DEFINE_string('log_dir', 'logs', 'Where to save training models')
flags.DEFINE_string('model_dir', None, 'Where model will be saved -- filled automatically')
flags.DEFINE_integer('log_img_step', 400, 'How often to visualize img during training')
flags.DEFINE_integer('epoch', 100, '# of epochs to train')


# Model config
flags.DEFINE_boolean(
    'smpl_prior', False,
    'if set, no adversarial prior is trained = monsters')

flags.DEFINE_boolean(
    'img_prior', False,
    'if set, no adversarial prior is trained = monsters')
flags.DEFINE_boolean(
    'use_swap_uv', False,
'if set, no adversarial prior is trained = monsters')
flags.DEFINE_boolean(
    'use_2d_joints', False,
    'if set, no adversarial prior is trained = monsters')
flags.DEFINE_boolean(
    'use_3d_joints', False,
    'if set, no adversarial prior is trained = monsters')
flags.DEFINE_boolean(
    'use_rotation', False,
'if set, no adversarial prior is trained = monsters')
# flags.DEFINE_integer('use_cam_batch', 4, 'how many views to use')

# Hyper parameters:
flags.DEFINE_float('e_lr', 0.001, 'Encoder learning rate')
flags.DEFINE_float('d_smpl_lr', 0.001, 'Adversarial prior learning rate')
flags.DEFINE_float('d_img_lr', 0.001, 'Adversarial prior learning rate')
flags.DEFINE_float('e_wd', 0.0001, 'Encoder weight decay')
flags.DEFINE_float('d_smpl_wd', 0.0001, 'Adversarial prior weight decay')
flags.DEFINE_float('d_img_wd', 0.0001, 'Adversarial prior weight decay')

flags.DEFINE_float('e_loss_weight', 1, 'weight on E_kp losses')
flags.DEFINE_float('e_feature_loss_weight', 2, 'weight on E_kp losses')
flags.DEFINE_float('d_smpl_loss_weight', 1, 'weight on discriminator')
flags.DEFINE_float('d_img_loss_weight', 1, 'weight on discriminator')
flags.DEFINE_float('e_j2d_weight', 1, 'weight on discriminator')
flags.DEFINE_float('e_j3d_weight', 1, 'weight on discriminator')
flags.DEFINE_float('recons_weight', 10., 'Encoder learning rate')

flags.DEFINE_boolean('use_test_transfer_image', False, 'finetune on test images')


def get_config():
    config = flags.FLAGS
    config(sys.argv)
    #########################train config################################
    # setattr(config,'pretrained_model_path','/home/chenf/PycharmProjects/multi-view-4d-fusion-master/models/model.ckpt-667589')
    setattr(config, 'pretrained_model_path',
            '/home/chenf/PycharmProjects/multi-view-4d-fusion-master/pretrain/model.ckpt-31000')
    setattr(config, 'feature_loss_pretrained_model_path','/home/chenf/PycharmProjects/multi-view-4d-fusion-master/models/model.ckpt')
    # setattr(config, 'color_pretrained_model_path','/home/gzp/research/2018/experiments/Human-pose-and-shape-transfer/logs/lyq/model.ckpt-19800')

    # setattr(config, 'model_dir', '/home/chenf/PycharmProjects/multi-view-4d-fusion-master/models/')
    setattr(config, 'e_lr', 1e-4)
    setattr(config, 'd_img_lr', 1e-4)
    setattr(config, 'color_length', 256)
    setattr(config, 'recons_weight', 80.)
    setattr(config, 'e_feature_loss_weight', 150.)
    setattr(config, 'log_img_step',100)
    setattr(config,'img_size',224)
    setattr(config, 'batch_size',20)
    setattr(config, 'epoch', 75)
    setattr(config, 'log_dir', 'lql')
    setattr(config, 'discriminator_type', 'stargan')
    # setattr(config, 'img_size', 448)
    setattr(config, 'use_swap_uv', True)
    setattr(config, 'use_2d_joints', False)
    setattr(config, 'use_3d_joints', False)
    setattr(config,'img_prior',True)
    # setattr(config, 'load_path', '/home/chenf/PycharmProjects/multi-view-4d-fusion-master/src/logs/HMR_Apr10_1520')
    # setattr(config, 'load_path', '/home/gzp/research/2018/experiments/Human-pose-and-shape-transfer/logs/HMR_Apr18_1222')
    # setattr(config, 'data_dir', '/home/chenf/Documents/pose_estimation/data/H36M-Multiview/train')
    setattr(config, 'data_dir', '/home/chenf/PycharmProjects/data/ExperimentData2/lql')
    setattr(config, 'use_test_transfer_image',True)
    ######################test config####################################
    # setattr(config, 'data_dir', '/home/chenf/Documents/pose_estimation/data/H36M-Multiview/test')
    # setattr(config, 'batch_size', 6)
    # setattr(config, 'load_path', '/home/gzp/research/2018/experiments/Human-pose-and-shape-transfer/logs/lyq')
    # setattr(config,'output_path','./lyq_pose_test')
    # setattr(config,'use_linear_pose',True)

    return config


# ----- For training ----- #


def prepare_dirs(config, prefix='HMR'):
    # Continue training from a load_path
    if config.load_path:
        if not osp.exists(config.load_path):
            print("load_path: %s doesnt exist..!!!" % config.load_path)
            import ipdb
            ipdb.set_trace()
        print('continuing from %s!' % config.load_path)

        # Check for changed training parameter:
        # Load prev config param path

        param_path = osp.join(config.load_path, 'params.json')
        # param_path = glob(osp.join(config.load_path, '*.json'))[0]

        with open(param_path, 'r') as fp:
            prev_config = json.load(fp)
        dict_here = config.__dict__
        ignore_keys = ['load_path', 'log_img_step', 'pretrained_model_path']
        diff_keys = [
            k for k in dict_here
            if k not in ignore_keys and k in prev_config.keys()
            and prev_config[k] != dict_here[k]
        ]

        for k in diff_keys:
            if k == 'load_path' or k == 'log_img_step':
                continue
            if prev_config[k] is None and dict_here[k] is not None:
                print("%s is different!! before: None after: %g" %
                      (k, dict_here[k]))
            elif prev_config[k] is not None and dict_here[k] is None:
                print("%s is different!! before: %g after: None" %
                      (k, prev_config[k]))
            else:
                print("%s is different!! before: " % k)
                print(prev_config[k])
                print("now:")
                print(dict_here[k])

        if len(diff_keys) > 0:
            print("really continue??")
            import ipdb
            ipdb.set_trace()

        config.model_dir = config.load_path

    else:

        # If config.dataset is not the same as default, add that to name.

        time_str = datetime.now().strftime("%b%d_%H%M")

        save_name = "%s_%s" % (prefix, time_str)
        config.model_dir = osp.join(config.log_dir, save_name)

    for path in [config.log_dir, config.model_dir]:
        if not osp.exists(path):
            print('making %s' % path)
            makedirs(path)


def save_config(config):
    param_path = osp.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    config_dict = {}
    for k in dir(config):
        config_dict[k] = config.__getattr__(k)

    with open(param_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=True)
