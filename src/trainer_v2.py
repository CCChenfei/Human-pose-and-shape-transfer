
from tensorflow.python.ops import control_flow_ops

from time import time
import tensorflow as tf
import numpy as np
import dirt
import os

from os.path import join, dirname
import deepdish as dd
from tf_smpl.batch_smpl import SMPL
from tf_smpl.projection import batch_orth_proj_idrot

from models.hmr_models import Color_Encoder_resnet, Color_Encoder_resnet18,Encoder_resnet, Encoder_fc3_dropout
from models.generator_models import InfoGenerator
from models.discriminator_models import NLayerDiscriminator, StarDiscriminator,MultiResDiscriminator
from models.resnet18 import ResNet
from utils.vis_utils import draw_skeleton

import cv2





class Fusion4DTrainer(object):
    def __init__(self, config, data_iter=None, real_data_iter=None, is_training=True,mocap_loader=None):
        self.config = config
        self.proj_fn = batch_orth_proj_idrot
        self.num_cam = 3
        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        self.g_global_step = tf.Variable(0, name='g_global_step', trainable=False)
        self.d_global_step = tf.Variable(0, name='d_global_step', trainable=False)
        # First make sure data_format is right
        self.data_iter = data_iter
        self.real_data_iter = real_data_iter
        self.mocap_loader = mocap_loader
        self.num_itr_per_epoch = self.config.num_images / self.config.batch_size
        self.feature_loss = ResNet()

        self.optimizer = tf.train.AdamOptimizer
        self.human_model = SMPL(self.config.smpl_model_path)
        self.human_uv = tf.constant(np.load(self.config.smpl_color_path)['global_uv'].astype(np.float32)) / 30.

        # For visualization:
        num2show = np.minimum(8, self.config.batch_size)
        # Take half from front & back
        self.show_these = tf.constant(
            np.hstack(
                [np.arange(num2show / 2), self.config.batch_size - np.arange(4) - 1]),
            tf.int32)
        self.show_swap_these = tf.constant(
            np.hstack(
                [np.arange(num2show / 2), 2*self.config.batch_size - np.arange(4) - 1]),
            tf.int32)
        self.is_training=is_training
        if self.is_training:
            self.build_model(self.is_training)
        else:
            self.build_test_model()

        # Logging
        self.init_fn = None
        if self.use_pretrained():
            # Make custom init_fn
            print("Fine-tuning from %s" % self.config.pretrained_model_path)

            self.pre_train_saver = tf.train.Saver(self.hmr_Var+self.color_Var + self.gen_Var)
            # self.color_pre_train_saver = tf.train.Saver(self.color_Var + self.gen_Var)
            self.feature_loss_saver = tf.train.Saver(self.resnet18_var)

            def load_pretrain(sess):
                self.pre_train_saver.restore(sess, self.config.pretrained_model_path)
                # self.color_pre_train_saver.restore(sess, self.config.color_pretrained_model_path)
                self.feature_loss_saver.restore(sess, self.config.feature_loss_pretrained_model_path)

            self.init_fn = load_pretrain
        if self.use_continued():
            # Make custom init_fn
            print("Continuing from %s" % self.config.model_dir)

            self.pre_train_saver = tf.train.Saver()

            def load_continued(sess):
                self.pre_train_saver.restore(sess, tf.train.latest_checkpoint(self.config.model_dir))

            self.init_fn = load_continued

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
        self.summary_writer = tf.summary.FileWriter(self.config.model_dir)
        self.save_name = self.summary_writer.get_logdir()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_options)
        self.session = tf.Session(config=self.sess_config)

    def use_pretrained(self):
        """
        Returns true only if:
          1. model_type is "resnet"
          2. pretrained_model_path is not None
          3. model_dir is NOT empty, meaning we're picking up from previous
             so fuck this pretrained model.
        """
        if (self.config.pretrained_model_path is
                not None):
            # Check is model_dir is empty
            import os
            if os.listdir(self.config.model_dir) == []:
                return True

        return False

    def use_continued(self):
        """
        Returns true only if:
          1. model_type is "resnet"
          2. pretrained_model_path is not None
          3. model_dir is NOT empty, meaning we're picking up from previous
             so fuck this pretrained model.
        """
        # Check is model_dir is empty
        import os
        if os.listdir(self.config.model_dir) != []:
            return True

        return False

    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        # Initialize scale at 0.9
        mean[0, 0] = 0.9
        mean_path = join(
            dirname(self.config.smpl_model_path), 'neutral_smpl_mean_params.h5')
        mean_vals = dd.io.load(mean_path)

        mean_pose = mean_vals['pose']
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        mean_shape = mean_vals['shape']

        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        mean[0, 3:] = np.hstack((mean_pose, mean_shape))
        mean = tf.constant(mean, tf.float32)
        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=True)
        self.hmr_Var.append(self.mean_var)
        init_mean = tf.tile(self.mean_var, [self.config.batch_size, 1])
        return init_mean


    def build_hmr_model(self,images,is_training,name=None):
        with tf.name_scope(name) as scope:
            #hmr's result have [cam,pose,shape]
            images=tf.image.resize_bilinear(images,(224,224))
            img_feat, self.hmr_Var = Encoder_resnet(images[:self.config.batch_size],
                                                       weight_decay=self.config.e_wd,
                                                       is_training=False,
                                                       reuse=tf.AUTO_REUSE)
            # regress smpl pars
            theta_prev = self.load_mean_param()

            # For visualizations
            show_all_verts = []
            show_all_pred_kps = []
            show_all_pred_cams = []


            # Main IEF loop
            for i in np.arange(self.config.num_stage):
                print('Iteration %d' % i)
                # ---- Compute outputs
                state = tf.concat([img_feat, theta_prev], 1)

                if i == 0:
                    delta_theta, threeD_var = Encoder_fc3_dropout(
                        state,
                        num_output=self.total_params,
                        reuse=tf.AUTO_REUSE)
                    self.hmr_Var.extend(threeD_var)
                else:
                    delta_theta, _ = Encoder_fc3_dropout(
                        state, num_output=self.total_params, reuse=tf.AUTO_REUSE)

                # Compute new theta
                theta_here = theta_prev + delta_theta
                # cam = N x 3, pose N x self.num_theta, shape: N x 10
                cams = theta_here[:, :self.config.num_cam]
                poses = theta_here[:, self.config.num_cam:(self.config.num_cam + self.config.num_theta)]
                shapes = theta_here[:, (self.config.num_cam + self.config.num_theta):]
                # Rs_wglobal is Nx24x3x3 rotation matrices of poses
                verts, Js, pred_Rs = self.human_model(shapes, poses, get_skin=True)
                pred_kp = batch_orth_proj_idrot(
                    Js, cams, name='proj2d_stage%d' % i)

                # Save things for visualiations:
                show_all_verts.append(tf.gather(verts, self.show_these))
                show_all_pred_kps.append(tf.gather(pred_kp, self.show_these))
                show_all_pred_cams.append(tf.gather(cams, self.show_these))

                # Finally update to end iteration.
                theta_prev = theta_here

            result={
                'pose':poses,
                'shape':shapes,
                'cam':cams,
            }
            return result,self.hmr_Var

    def build_color_encoder(self,images,uvs,is_training):
        color_feat, color_Var = Color_Encoder_resnet(
            tf.concat([images, uvs], axis=-1),
            weight_decay=self.config.e_wd,
            reuse=tf.AUTO_REUSE,
            is_training=is_training)

        color_code, color_fc_Var = Encoder_fc3_dropout(color_feat,
                                                       num_output=self.config.color_length,
                                                       reuse=tf.AUTO_REUSE,
                                                       name='color_stage')
        color_Var.extend(color_fc_Var)

        return color_code,color_Var

    def hmr2uv(self,hmr_result):
        #Js is 3D joints,has cocoplus format:shape=(19,3),
        # first 14 are body joints and last 5 are face points

        verts, Js,_ = self.human_model(hmr_result['shape'],hmr_result['pose'],get_skin=True)
        verts_2d = self.proj_fn(verts, hmr_result['cam'])


        #normalized coordinate
        verts_2d_homo=tf.concat([verts_2d,
                   (3. + verts[:, :, 2:3]) / tf.expand_dims(
                       tf.expand_dims(tf.reduce_max(3. + verts[:, :, 2], axis=-1), axis=-1),
                       axis=-1),
                   tf.ones_like(verts_2d[:, :, 0:1])],
                  axis=-1)
        batch_size=verts.shape[0].value
        #use dirt to render smpl verts,the color is UV
        smpl_uv = dirt.rasterise_batch(
            background=tf.zeros(shape=[batch_size,self.config.img_size,self.config.img_size,3]),
            vertices=verts_2d_homo,
            faces=tf.stack([self.human_model.faces] * batch_size, axis=0),
            vertex_colors=tf.stack(
                [tf.concat([self.human_uv, tf.zeros_like(self.human_uv[:, 0:1])], axis=-1)] * batch_size,
                axis=0),
            height=self.config.img_size,
            width=self.config.img_size,
            channels=3,
            name='renderer')

        smpl_uv = tf.reverse(smpl_uv[:, :, :, :2], axis=[1])
        return smpl_uv

    def build_single_generator(self,color_code,uv,bg,is_training):
        mask, fore, var_generator = InfoGenerator(uv, color_code,is_training=is_training,name='generator',reuse=tf.AUTO_REUSE)
        img = (1. - mask) * tf.image.resize_bilinear(bg,[256, 256]) + mask * fore
        # final_img = tf.image.resize_bilinear(img, [self.config.img_size, self.config.img_size])
        final_img = img #increase the resolution of the image 512*512*3
        output={
            'mask':mask,
            'fore':fore,
            'img':final_img,
        }
        return output,var_generator


    def build_generator(self,input_feature,is_training):
        #data process
        self.image_loader=input_feature["image"]
        self.bg_loader = input_feature["bg"]
        #use joints gt to constrain HMR model when HMR updates
        if self.config.use_2d_joints:
            self.j2d_loader = input_feature["j2d"]
        if self.config.use_3d_joints:
            self.j3d_loader = input_feature["j3d"]
        if self.config.use_rotation:
            self.rotation_loader=input_feature["rot"]


        self.smpl_result,self.hmr_Var=self.build_hmr_model(self.image_loader,is_training)
        self.smpl_uv=self.hmr2uv(self.smpl_result)
        self.color_code,self.color_Var=self.build_color_encoder(self.image_loader[:self.config.batch_size],self.smpl_uv,is_training)

        self.output,self.gen_Var=self.build_single_generator(self.color_code,self.smpl_uv,self.bg_loader,is_training)
        if self.config.use_swap_uv:
            self.color_code_swap = tf.reshape(self.color_code, [self.config.batch_size / 2, 2, -1])
            self.color_code_swap = tf.reshape(tf.stack([self.color_code_swap[:, 1, :], self.color_code_swap[:, 0, :]], axis=1),
                                         [self.config.batch_size, -1])
            self.output_swap,_=self.build_single_generator(self.color_code_swap,self.smpl_uv,self.bg_loader,is_training)
            for key in self.output.keys():
                self.output[key]=tf.concat([
                    self.output[key],
                    self.output_swap[key]
                ],axis=0)


        return self.output['img']


    def build_discriminator(self,input_feature):
        if self.config.discriminator_type == 'patchgan':
            self.discriminator = NLayerDiscriminator
        elif self.config.discriminator_type=='stargan':
            self.discriminator = StarDiscriminator
        elif self.config.discriminator_type=='multires':
            self.discriminator=MultiResDiscriminator
        else:
            raise NotImplementedError

        logits,dimg_Var=self.discriminator(x=input_feature,name='image_dis',reuse=tf.AUTO_REUSE)

        return logits,dimg_Var


    def build_model(self,is_training=True):

        self.input_feature={
            'image':tf.placeholder(dtype=tf.float32,
                                               shape=[self.config.batch_size, self.config.img_size,
                                                      self.config.img_size, 3]),
            'bg':tf.placeholder(dtype=tf.float32,
                                               shape=[self.config.batch_size, self.config.img_size,
                                                      self.config.img_size, 3]),
            'real': tf.placeholder(dtype=tf.float32,
                                   shape=[self.config.batch_size, self.config.img_size,
                                          self.config.img_size, 3]),
        }
        if self.config.use_swap_uv:
            real_size=2
        else:
            real_size=1


        if self.config.use_2d_joints:
            self.input_feature.update({
                'j2d':tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.joints_num, 2])
            })
        if self.config.use_3d_joints:
            self.input_feature.update({
                'j3d':tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.joints_num, 3])
            })
        if self.config.use_rotation:
            self.input_feature.update({
                'rot': tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, 3, 3])
            })

        self.output_gen=self.build_generator(self.input_feature,is_training)
        self.output_gen_assign=tf.Variable(tf.zeros_like(self.output_gen,dtype=tf.float32),trainable=False,name='output_assign')
        self.assign_output_op = tf.assign(self.output_gen_assign, self.output_gen)


        self.fake_score,_=self.build_discriminator(self.output_gen)
        self.fake_score_assign,_=self.build_discriminator(self.output_gen_assign)
        # self.real_score,self.dimg_Var = self.build_discriminator(self.input_feature['real'])
        self.real_score,self.dimg_Var = self.build_discriminator(tf.image.resize_bilinear(self.input_feature['real'],[256,256]))

        #gather generator loss
        # gather reconstruction loss
        # fake_feature_list, self.resnet18_var = self.feature_loss.build_tower(
        #     tf.image.resize_bilinear(self.move_mean_and_var(self.output_gen), [128, 128])
        #     , reuse=False)
        # real_feature_list, _ = self.feature_loss.build_tower(
        #     tf.image.resize_bilinear(self.move_mean_and_var(self.input_feature['image']), [128, 128])
        #     , reuse=True)
        fake_feature_list, self.resnet18_var = self.feature_loss.build_tower(
            tf.image.resize_bilinear(self.move_mean_and_var(self.output_gen), [256, 256])
            , reuse=False)
        real_feature_list, _ = self.feature_loss.build_tower(
            tf.image.resize_bilinear(self.move_mean_and_var(self.input_feature['image']), [256, 256])
            , reuse=True)
        if self.config.use_swap_uv:
            for i in range(len(real_feature_list)):
                real_feature_list[i]=tf.concat([real_feature_list[i]]*2,axis=0)
        feature_loss_list = [tf.reduce_mean(tf.abs(real_feature_list[i] - fake_feature_list[i])) for i in
                             range(len(fake_feature_list))]
        self.feature_loss = tf.reduce_mean(tf.stack(feature_loss_list, axis=0))
        if self.config.use_swap_uv:
            # self.loss_recons = tf.reduce_mean((self.output_gen -
            #                                tf.concat([self.input_feature['image']]*2,axis=0)) ** 2,name='ggg')
            self.loss_recons = tf.reduce_mean(tf.abs(self.output_gen -
                                               tf.concat([tf.image.resize_bilinear(self.input_feature['image'],[256,256])] * 2, axis=0)), name='ggg')
        else:
            # self.loss_recons = tf.reduce_mean((self.output_gen -
            #                                    self.input_feature['image']) ** 2)
            self.loss_recons = tf.reduce_mean(tf.abs(self.output_gen -
                                               tf.image.resize_bilinear(self.input_feature['image'],[256,256])))
        self.e_loss = self.config.recons_weight * self.loss_recons \
                      + self.config.e_feature_loss_weight * self.feature_loss

        self.e_img_loss_disc = tf.reduce_mean(
            -self.fake_score)
        if self.config.img_prior:
            self.e_loss += self.config.d_img_loss_weight * self.e_img_loss_disc

        # gather discriminator loss
        def gradient_penalty(real, fake):
            def interpolate(a, b):
                shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
                alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.get_shape().as_list())
                return inter

            x = interpolate(real, fake)
            pred, _ = self.discriminator(name='image_dis', x=x, reuse=tf.AUTO_REUSE)
            gradients = tf.gradients(pred, x)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
            gp = tf.reduce_mean((slopes - 1.) ** 2)
            return gp

        if self.config.use_swap_uv:
            self.gp = gradient_penalty(tf.concat([
                tf.image.resize_bilinear(self.input_feature['real'],[256,256])]*2,axis=0),self.output_gen_assign)

        else:
            self.gp = gradient_penalty(tf.image.resize_bilinear(self.input_feature['real'],[256,256]),
                                       self.output_gen_assign)

        self.d_img_loss_real = tf.reduce_mean(
            -self.real_score)
        self.d_img_loss_fake = tf.reduce_mean(
            self.fake_score_assign)
        self.d_img_loss = self.config.d_img_loss_weight * (
            self.d_img_loss_fake+self.d_img_loss_real+10.0*self.gp)



        print('collecting batch norm moving means!!')
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            self.e_loss = control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.e_loss)
            # self.d_img_loss = control_flow_ops.with_dependencies(
            #     [tf.group(*bn_ops)], self.d_img_loss)

        # self.e_lr=tf.train.exponential_decay(self.config.e_lr,global_step=self.g_global_step,
        #                                      decay_steps=200,decay_rate=0.96)
        self.e_lr=tf.train.polynomial_decay(10*self.config.e_lr,self.g_global_step,10000,self.config.e_lr)
        e_optimizer = self.optimizer(self.e_lr)
        self.e_opt = e_optimizer.minimize(
            self.e_loss, global_step=self.g_global_step, var_list=self.color_Var + self.gen_Var)

        d_img_optimizer = self.optimizer(self.config.d_img_lr)
        self.d_img_opt = d_img_optimizer.minimize(
            self.d_img_loss, global_step=self.d_global_step, var_list=self.dimg_Var)

        self.setup_summaries()

    def build_test_model(self):
        self.input_feature = {
            'image': tf.placeholder(dtype=tf.float32,
                                    shape=[self.config.batch_size, self.config.img_size,
                                           self.config.img_size, 3]),
            'bg': tf.placeholder(dtype=tf.float32,
                                 shape=[self.config.batch_size, self.config.img_size,
                                        self.config.img_size, 3]),
        }

        self.input_feature_single = {
            'image': tf.placeholder(dtype=tf.float32,
                                    shape=[1, self.config.img_size,
                                           self.config.img_size, 3]),
            'bg': tf.placeholder(dtype=tf.float32,
                                 shape=[1, self.config.img_size,
                                        self.config.img_size, 3]),
        }

        self.smpl_place={
            'cam':tf.placeholder(dtype=tf.float32,
                                 shape=[1,self.config.num_cam]),
            'pose':tf.placeholder(dtype=tf.float32,
                                 shape=[1,self.num_theta]),
            'shape':tf.placeholder(dtype=tf.float32,
                                 shape=[1,10])
        }

        self.smpl_place_rand = {
            'cam': tf.placeholder(dtype=tf.float32,
                                  shape=[self.config.rand_batch_num, self.config.num_cam]),
            'pose': tf.placeholder(dtype=tf.float32,
                                   shape=[self.config.rand_batch_num, self.num_theta]),
            'shape': tf.placeholder(dtype=tf.float32,
                                    shape=[self.config.rand_batch_num, 10])
        }

        self.smpl_result, self.hmr_Var = self.build_hmr_model(self.input_feature['image'], is_training=False)



        self.smpl_uv = self.hmr2uv(self.smpl_place)
        self.smpl_uv_rand=self.hmr2uv(self.smpl_place_rand)

        self.color_code, self.color_Var = self.build_color_encoder(self.input_feature_single['image'],
                                                                   self.smpl_uv, is_training=False)

        self.output, self.gen_Var = self.build_single_generator(tf.concat([self.color_code]*self.config.rand_batch_num,axis=0),
                                                                self.smpl_uv_rand,
                                                                tf.concat(
                                                                    [ self.input_feature_single['bg']] * self.config.rand_batch_num,
                                                                    axis=0),

                                                                is_training=False)



    def mkdir(self,name):
        if not os.path.exists(name):
            os.mkdir(name)

    def init_test(self):
        self.session.run(tf.global_variables_initializer())
        if self.init_fn is not None:
            self.init_fn(self.session)

    def rot_pose(self,pose,t):
        R_t=cv2.Rodrigues(t)[0]
        R_pose=cv2.Rodrigues(pose)[0]
        R_total=R_t.dot(R_pose)
        return cv2.Rodrigues(R_total)[0]


    def test(self,image,bg):

        fetch_dict={
            'cam':self.smpl_result['cam'],
            'pose':self.smpl_result['pose'],
            'shape':self.smpl_result['shape'],
        }
        feed_dict={
            self.input_feature['image']:image,
        }
        #get smpl and camera pars from image using hmr net
        smpl_result_np=self.session.run(fetch_dict,feed_dict)

        #randomize smpl and camera pars
        # if self.config.use_random_cam:
        #     rand_cam=np.expand_dims(smpl_result_np['cam'],axis=0)+np.random.normal(loc=0.,scale=self.config.cam_scale,size=[self.config.rand_batch_num]+list(smpl_result_np['cam'].shape))
        #     rand_cam=np.transpose(rand_cam,[1,0,2])
        #
        # else:
        #     rand_cam=np.stack([smpl_result_np['cam']]*self.config.rand_batch_num,axis=1)
        rand_cam = np.stack([smpl_result_np['cam']] * self.config.rand_batch_num, axis=1)
        if self.config.use_random_pose:
            rand_pose = np.expand_dims(smpl_result_np['pose'], axis=0) + np.random.normal(loc=0.,
                                                                    scale=self.config.pose_scale,
                                                                                        size=[
                                                                                                 self.config.rand_batch_num] + list(
                                                                                            smpl_result_np[
                                                                                                'pose'].shape))
            rand_pose[:,:,:3]=np.stack([smpl_result_np['pose'][:,:3]]*self.config.rand_batch_num,axis=0)
            rand_pose = np.transpose(rand_pose, [1, 0, 2])

        elif self.config.use_linear_pose:
            rand_pose = []
            pose_mean=np.zeros_like(smpl_result_np['pose'])
            for i in range(self.config.rand_batch_num):
                rand_pose.append((smpl_result_np['pose']-pose_mean) * (i + 1) / float(self.config.rand_batch_num)+pose_mean)
            rand_pose = np.stack(rand_pose, axis=1)
            rand_pose[:, :, :3] = np.stack([smpl_result_np['pose'][:, :3]] * self.config.rand_batch_num, axis=1)
        else:
            rand_pose = np.stack([smpl_result_np['pose']] * self.config.rand_batch_num, axis=1)



        if self.config.use_random_cam:
            rand_pose_rot=[]
            for i in range(self.config.batch_size):
                pose_i=smpl_result_np['pose'][i,:3]
                for j in range(self.config.rand_batch_num):
                    root_j=np.array([0,1,0.])*np.random.normal(0.,self.config.cam_scale)
                    rand_pose_rot.append(self.rot_pose(pose_i,root_j))
            rand_pose_rot=np.stack(rand_pose_rot,axis=0).reshape((-1,self.config.rand_batch_num,3))
            rand_pose[:,:,:3]=rand_pose_rot
        elif self.config.use_linear_cam:
            rand_pose_rot = []

            for i in range(self.config.batch_size):
                pose_i = smpl_result_np['pose'][i, :3]
                for j in range(self.config.rand_batch_num):
                    root_j = np.array([0, 1, 0.])*(float(j)/self.config.rand_batch_num)*2*np.pi
                    rand_pose_rot.append(self.rot_pose(pose_i, root_j))
            rand_pose_rot = np.stack(rand_pose_rot, axis=0).reshape((self.config.batch_size,self.config.rand_batch_num, 3))
            rand_pose[:,:, :3] = rand_pose_rot


        if self.config.use_random_shape:
            rand_shape = np.expand_dims(smpl_result_np['shape'], axis=0) + np.random.normal(loc=0.,
                                                                                        scale=self.config.shape_scale,
                                                                                        size=[
                                                                                                 self.config.rand_batch_num] + list(
                                                                                            smpl_result_np[
                                                                                                'shape'].shape))
            rand_shape = np.transpose(rand_shape, [1, 0, 2])
        elif self.config.use_linear_shape:
            rand_shape=[]
            shape_mean=np.zeros_like(smpl_result_np['shape'])
            shape_mean[:,1]=3.
            for i in range(self.config.rand_batch_num):
                rand_shape.append((smpl_result_np['shape']-shape_mean)*(i+1)/float(self.config.rand_batch_num)+shape_mean)
            rand_shape=np.stack(rand_shape,axis=1)
        else:
            rand_shape = np.stack([smpl_result_np['shape']] * self.config.rand_batch_num, axis=1)

        if self.config.use_random_cam or self.config.use_random_pose or self.config.use_random_shape or self.config.use_linear_shape or self.config.use_linear_pose or self.config.use_linear_cam:
            smpl_result_rand={
                'cam':rand_cam,
                'pose':rand_pose,
                'shape':rand_shape
            }
        else:
            smpl_result_rand={
                'cam': np.expand_dims(smpl_result_np['cam'],axis=1),
                'pose': np.expand_dims(smpl_result_np['pose'],axis=1),
                'shape': np.expand_dims(smpl_result_np['shape'],axis=1)
            }

        result={
            'uv':[],
            'uv_rand':[],
            'mask':[],
            'fore':[],
            'img':[]
        }

        for i in range(self.config.batch_size):
            feed_dict={
                self.input_feature_single['image']:np.expand_dims(image[i],axis=0),
                self.input_feature_single['bg']:np.expand_dims(bg[i],axis=0),
                self.smpl_place['cam']:np.expand_dims(smpl_result_np['cam'][i],axis=0),
                self.smpl_place['pose']: np.expand_dims(smpl_result_np['pose'][i], axis=0),
                self.smpl_place['shape']: np.expand_dims(smpl_result_np['shape'][i], axis=0),

                self.smpl_place_rand['cam']: smpl_result_rand['cam'][i],
                self.smpl_place_rand['pose']: smpl_result_rand['pose'][i],
                self.smpl_place_rand['shape']: smpl_result_rand['shape'][i],
            }

            fetch_dict=self.output
            fetch_dict.update({
                'uv_rand':self.smpl_uv_rand,
                'uv':self.smpl_uv
            })
            output=self.session.run(fetch_dict,feed_dict)
            for key in result.keys():
                result[key].append(output[key])

        for key in result.keys():
            result[key]=np.stack(result[key],axis=0)
        return result


    def train(self):
        # For rendering!
        step = 0
        self.session.run(tf.global_variables_initializer())
        if self.init_fn is not None:
            self.init_fn(self.session)

        def update_fetch(state,step):
            if state=='g':
                fetch_dict = {
                    # The meat
                    "loss_recons": self.loss_recons,
                    "e_loss": self.e_loss,
                    "e_opt": self.e_opt,
                    "step": self.g_global_step,
                    "output":self.output_gen,
                    "assign_op":self.assign_output_op,
                    "summary":self.generator_summary_op_always,
                }
                if self.config.img_prior:
                    fetch_dict.update({
                        "loss_img_disc": self.e_img_loss_disc,
                    })
                if step%self.config.log_img_step==0:
                    fetch_dict.update({
                        'show_total_image':self.show_total_image
                    })
                    if self.config.use_2d_joints:
                        fetch_dict.update({
                            'show_input': self.show_input,
                            'show_output':self.show_out,
                        })


            else:
                fetch_dict={
                    # For D:

                    "d_img_opt": self.d_img_opt,
                    "d_img_loss": self.d_img_loss,
                    "summary": self.dis_summary_op_always,
                    "step":self.g_global_step,
                }
            return fetch_dict

        def update_feed(state):
            feed_dict={}

            if state=='g':
                if self.config.use_2d_joints or self.config.use_3d_joints:
                    img, bg, R, ID, D2, D3 = next(self.data_iter)
                else:
                    img, bg, R, ID = next(self.data_iter)

                feed_dict = {

                    self.input_feature['image']: img[:, 0, :, :, :],
                    self.input_feature['bg']: bg[:, 0, :, :, :],
                }
                if self.config.use_2d_joints:
                    feed_dict.update({
                        self.input_feature['j2d']: D2[:, 0, :, :],
                    })
                if self.config.use_3d_joints:
                    feed_dict.update({
                        self.input_feature['j3d']: D3[:, 0, :, :],
                    })
                if self.config.use_rotation:
                    feed_dict.update({
                        self.input_feature['rot']: R[:, 0, :, :],
                    })

            if state=='d':

                real_img, real_bg, real_R, real_ID = next(self.real_data_iter)
                feed_dict={

                    self.input_feature['real']: real_img[:, 0, :, :, :],
                }

            return feed_dict

        while True:
            t0 = time()

            if self.config.img_prior:
                if step!=0:
                    for i in range(5):

                        self.result = self.session.run(update_fetch('d',step), feed_dict=update_feed('d'))
                    self.summary_writer.add_summary(self.result['summary'],global_step=self.result['step'])

            self.result = self.session.run(update_fetch('g',step), feed_dict=update_feed('g'))
            self.summary_writer.add_summary(self.result['summary'], global_step=self.result['step'])
            t1 = time()
            result=self.result

            e_loss = result['e_loss']
            step = result['step']

            epoch = float(step) / self.num_itr_per_epoch

            print("itr %d/(epoch %.1f): time %g, Enc_loss: %.4f" %
                  (step, epoch, t1 - t0, e_loss))

            if step % 300 == 0:
                self.saver.save(self.session, os.path.join(self.save_name, 'model.ckpt'), global_step=result['step'])

            if step % self.config.log_img_step == 0:
                self.draw_results(result)

            self.summary_writer.flush()
            if epoch > self.config.epoch:
                break

            step += 1

        print('Finish training on %s' % self.config.model_dir)


    def visualize_img(self, input, output, gt_kp, pred_kp):
        """
        Overlays gt_kp and pred_kp on img.
        Draws vert with text.
        Renderer is an instance of SMPLRenderer.
        """

        # Draw skeleton

        gt_joint = np.transpose(((gt_kp[:, :, :2] + 1) * 0.5) * self.config.img_size, [0, 2, 1])
        pred_joint = np.transpose(((pred_kp + 1) * 0.5) * self.config.img_size, [0, 2, 1])
        input_ = []
        output_ = []
        for i in range(input.shape[0]):
            input_with_gt = draw_skeleton(
                input[i], gt_joint[i], draw_edges=False)
            # import ipdb
            # ipdb.set_trace()
            input_with_skel = draw_skeleton(input_with_gt, pred_joint[i])
            input_.append(input_with_skel)

            output_with_gt = draw_skeleton(
                output[i], gt_joint[i], draw_edges=False)
            output_with_skel = draw_skeleton(output_with_gt, pred_joint[i])
            output_.append(output_with_skel)
        input_ = np.stack(input_, axis=0)
        output_ = np.stack(output_, axis=0)
        # import ipdb
        # ipdb.set_trace()

        combined = np.concatenate([input_, output_], axis=-2)
        return combined

    def draw_results(self, result):

        from StringIO import StringIO
        import matplotlib.pyplot as plt
        show_total_image = result['show_total_image']
        if self.config.use_2d_joints:
            show_kps_image = self.visualize_img(result["show_input"], result["show_out"],
                                                result["show_kps_gt"], result["show_kps"])
            show_kps_image = np.stack([show_kps_image[:, :, :, 2], show_kps_image[:, :, :, 1],
                                       show_kps_image[:, :, :, 0]], axis=-1)
        img_summaries = []

        for i in range(show_total_image.shape[0]):
            sio = StringIO()
            if self.config.use_2d_joints:
                new = (np.concatenate([show_total_image[i], show_kps_image[i]], axis=0)).astype(np.uint8)
            else:
                new = show_total_image[i].astype(np.uint8)
            plt.imsave(sio, new, format='png')
            vis_sum = tf.Summary.Image(
                encoded_image_string=sio.getvalue(),
                height=new.shape[0],
                width=new.shape[1])
            img_summaries.append(
                tf.Summary.Value(tag="vis_images/%d" % i, image=vis_sum))

        img_summary = tf.Summary(value=img_summaries)
        self.summary_writer.add_summary(
            img_summary, global_step=result['step'])


    def move_mean_and_var(self, x):
        img_mean = tf.constant([0.485, 0.456, 0.406])
        img_mean = tf.expand_dims(img_mean, axis=0)
        img_mean = tf.expand_dims(img_mean, axis=0)
        img_mean = tf.expand_dims(img_mean, axis=0)

        img_std = tf.constant([0.229, 0.224, 0.225])
        img_std = tf.expand_dims(img_std, axis=0)
        img_std = tf.expand_dims(img_std, axis=0)
        img_std = tf.expand_dims(img_std, axis=0)

        return ((x + 1) / 2. - img_mean) / img_std

    def setup_summaries(self):
        generator_always_report = [
            tf.summary.scalar("loss/loss_recons", self.loss_recons),
            tf.summary.scalar('loss/loss_feature', self.feature_loss),
            tf.summary.scalar('loss/e_loss', self.e_loss),
            tf.summary.scalar("loss/e_img_loss_disc", self.e_img_loss_disc),

        ]
        if self.config.use_2d_joints:
            generator_always_report.extend([
                tf.summary.scalar("loss/loss_2d_joints", self.loss_2d_joints),
            ])
        if self.config.use_3d_joints:
            generator_always_report.extend([
                tf.summary.scalar("loss/loss_3d_joints", self.loss_3d_joints),
            ])


        if self.config.img_prior:
            dis_always_report=[
                tf.summary.scalar("loss/d_img_loss", self.d_img_loss),
                tf.summary.scalar("loss/d_img_loss_real", self.d_img_loss_real),
                tf.summary.scalar("loss/d_img_loss_fake", self.d_img_loss_fake),

            ]
            if self.config.discriminator_loss_type == 'began':
                dis_always_report.extend([
                    tf.summary.scalar("loss/k_t", self.k_t),
                    tf.summary.scalar("loss/balance", self.balance),

                ])
            elif self.config.discriminator_loss_type == 'wgan':
                dis_always_report.extend([
                    tf.summary.scalar('loss/d_gp', self.gp),
                ])
            self.dis_summary_op_always=tf.summary.merge(dis_always_report)


        self.generator_summary_op_always = tf.summary.merge(generator_always_report)

        input_image = (tf.gather(self.input_feature['image'], self.show_these) + 1.) * 255. / 2.
        bg_image = (tf.gather(self.input_feature['bg'], self.show_these) + 1.) * 255. / 2.
        u_image = (tf.stack([tf.gather(self.smpl_uv, self.show_these)[:, :, :, 0]] * 3, -1)) * 255.
        v_image = (tf.stack([tf.gather(self.smpl_uv, self.show_these)[:, :, :, 1]] * 3, -1)) * 255.
        if self.config.use_swap_uv:
            show=self.show_swap_these
        else:
            show=self.show_these
        mask = (tf.concat([tf.gather(self.output['mask'], show)] * 3, -1)) * 255.
        fore = (tf.gather(self.output['fore'], show) + 1.) * 255. / 2.
        final = (tf.gather(self.output['img'], show) + 1.) * 255. / 2.
        ####show low resolution image####
        final= tf.image.resize_bilinear(final,[self.config.img_size,self.config.img_size])

        dis = tf.abs(final - input_image)
        total_image = tf.concat([
            tf.concat(
                [input_image, u_image, tf.image.resize_bilinear(mask, [self.config.img_size, self.config.img_size]),
                 bg_image], axis=1),
            tf.concat(
                [final, v_image, tf.image.resize_bilinear(fore, [self.config.img_size, self.config.img_size]), dis],
                axis=1),
        ], axis=2)

        self.show_total_image = tf.stack([total_image[:, :, :, 2],
                                          total_image[:, :, :, 1],
                                          total_image[:, :, :, 0]], axis=-1)

        self.show_input = input_image
        self.show_out = final
        if self.config.use_2d_joints:
            self.show_kp_gt = tf.gather(self.j2d_gt, self.show_these)
            self.show_kp = tf.gather(self.final_kp[:, :14, :], self.show_these)




