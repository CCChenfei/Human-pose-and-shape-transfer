import tensorflow as tf

from config import get_config, prepare_dirs, flags
from dataloader.dataloader import DataLoader
from trainer_v2 import Fusion4DTrainer
import numpy as np
import os
import cv2
from utils.trans_video import trans_video

flags.DEFINE_boolean('use_random_pose',False,'use_random_pose')
flags.DEFINE_float('cam_scale',0.1,'cam_scale')
flags.DEFINE_boolean('use_random_cam',False,'use_random_cam')
flags.DEFINE_float('pose_scale',0.1,'pose_scale')
flags.DEFINE_boolean('use_random_shape',False,'use_random_shape')
flags.DEFINE_float('shape_scale',0.1,'shape_scale')
flags.DEFINE_boolean('use_linear_shape',False,'use linear shape')
flags.DEFINE_boolean('use_linear_pose',False,'use linear pose')
flags.DEFINE_boolean('use_linear_cam',False,'use linear cam')
flags.DEFINE_integer('rand_batch_num',70,'rand_batch_num')
flags.DEFINE_string('output_path',None,'output image path')



def denormalize_rgb(img):
    return ((img+1.)*255./2.).astype(np.uint8)

def denormalize_uv(uv):
    return (uv*255.).astype(np.uint8)


def mkdir(name):
    if not os.path.exists(name):
        os.mkdir(name)

def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config,randomize=False)
        data_iter = data_loader.generator(usePoseLabel=False,
                                          shuffle_app=False,
                                          cam_shuffle=False,
                                          )


    trainer = Fusion4DTrainer(config, is_training=False)
    #save_config(config)
    trainer.init_test()
    img, bg, R, ID=next(data_iter)
    img=img.reshape((config.batch_size,config.useCamBatches,config.img_size,config.img_size,3))
    bg = bg.reshape((config.batch_size,config.useCamBatches, config.img_size, config.img_size, 3))
    result=[]
    for i in range(config.useCamBatches):
        result.append(trainer.test(img[:,i],bg[:,i]))

    if config.output_path is not None:
        mkdir(config.output_path)
        for c in range(config.useCamBatches):
            for i in range(config.batch_size):
                global_name=os.path.join(config.output_path,'res_i%d_c%d'%(i,c))

                cv2.imwrite( global_name+'_input.png', denormalize_rgb(img[i,c]))
                cv2.imwrite(global_name + '_bg.png', denormalize_rgb(bg[i,c]))
                cv2.imwrite(global_name+'_real_u.png',denormalize_uv(result[c]['uv'][i,0,:,:,0]))
                cv2.imwrite(global_name + '_real_v.png', denormalize_uv(result[c]['uv'][i, 0, :, :, 1]))
                for j in range(config.rand_batch_num):
                    name=os.path.join(config.output_path,'res_i%d_c%d_r%d'%(i,c,j))
                    cv2.imwrite(name+'_final.png',denormalize_rgb(result[c]['img'][i,j]))
                    cv2.imwrite(name + '_mask.png', denormalize_uv(result[c]['mask'][i, j]))
                    cv2.imwrite(name + '_fore.png', denormalize_rgb(result[c]['fore'][i, j]))
                    cv2.imwrite(name + '_u.png', denormalize_uv(result[c]['uv_rand'][i, j,:,:,0]))
                    cv2.imwrite(name + '_v.png', denormalize_uv(result[c]['uv_rand'][i, j, :, :, 1]))

        if config.use_linear_pose or config.use_linear_shape or config.use_linear_cam:
            for i in range(config.batch_size):
                for j in range(config.useCamBatches):
                    save_name=os.path.join(config.output_path,'res_i%d_c%d.avi'%(i,j))
                    trans_video(config.output_path,save_name,i,j,config.rand_batch_num,config.img_size)
        print('Save Done!! In %s'%config.output_path)



if __name__ == '__main__':
    config = get_config()
    main(config)