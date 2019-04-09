from absl import flags
import sys
import os
import cv2
import numpy as np

# flags.DEFINE_string('input_dir','/home/gzp/research/2018/experiments/multi-view-4d-fusion/test1/','image dir')
# flags.DEFINE_integer('idx_i',1,'i')
# flags.DEFINE_integer('idx_c',0,'c')
# flags.DEFINE_integer('rand_batch_num',70,'rand batch num')
# flags.DEFINE_integer('img_size',224,'img size')
# flags.DEFINE_string('output','test.avi','output video name')
#
#
# def get_config():
#     config = flags.FLAGS
#     config(sys.argv)
#
#     # if 'resnet' in config.model_type:
#     #     setattr(config, 'img_size', 224)
#     #     # Slim resnet wants NHWC..
#     #     setattr(config, 'data_format', 'NHWC')
#
#     return config

def trans_video(input_dir,output,idx_i,idx_c,rand_batch_num=70,img_size=224):
    videoWriter = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('I', '4', '2', '0'), 10, (3*img_size, 2*img_size))
    for i in range(rand_batch_num-1,0,-1):
        name=os.path.join(input_dir,'res_i%d_c%d_r%d'%(idx_i,idx_c,i))
        final=cv2.imread(name+'_final.png')
        u=cv2.imread(name+'_u.png')
        mask=cv2.resize(cv2.imread(name+'_mask.png'),dsize=(img_size, img_size))
        fore=cv2.resize(cv2.imread(name+'_fore.png'),dsize=(img_size, img_size))
        input=cv2.imread(os.path.join(input_dir,'res_i%d_c%d_input.png'%(idx_i,idx_c)))
        bg=cv2.imread(os.path.join(input_dir,'res_i%d_c%d_bg.png'%(idx_i,idx_c)))
        total=np.concatenate([np.concatenate([input,final,u],axis=1),np.concatenate([bg,fore,mask],axis=1)],axis=0)
        videoWriter.write(total)
