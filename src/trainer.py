from tensorflow.python.ops import control_flow_ops

from time import time
import tensorflow as tf
import numpy as np
import dirt
import os

from os.path import join, dirname
import deepdish as dd

from tf_smpl.batch_lbs import batch_rodrigues,batch_inv_rodrigues,batch_worlds_relate
from tf_smpl.batch_smpl import SMPL
from tf_smpl.projection import batch_orth_proj_idrot

# from models.hmr_models import Encoder_resnet,Encoder_fc3_dropout,\
#     Discriminator_separable_rotations
from models.hmr_models import Color_Encoder_resnet,Encoder_resnet,Encoder_fc3_dropout,\
    Discriminator_separable_rotations
from models.generator_models import InfoGenerator
from models.discriminator_models import NLayerDiscriminator,DiscriminatorBEGAN,StarDiscriminator
# from models.discriminator_models import NLayerDiscriminator,DiscriminatorBEGAN
from utils.vis_utils import draw_skeleton,draw_text,plot_j3d
from models.resnet18 import ResNet
import cv2
class Fusion4DTrainer(object):
    def __init__(self, config, data_iter,real_data_iter=None,mocap_loader=None):
        self.config = config
        self.proj_fn = batch_orth_proj_idrot
        self.num_cam = 3
        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.g_global_step = tf.Variable(0, name='g_global_step', trainable=False)
        # self.d_global_step = tf.Variable(0, name='d_global_step', trainable=False)
        # First make sure data_format is right
        self.data_iter=data_iter
        self.real_data_iter=real_data_iter
        self.mocap_loader=mocap_loader
        self.num_itr_per_epoch=self.config.num_images/self.config.batch_size
        self.feature_loss = ResNet()

        self.optimizer = tf.train.AdamOptimizer
        self.human_model = SMPL(self.config.smpl_model_path)
        self.human_uv=tf.constant(np.load(self.config.smpl_color_path)['global_uv'].astype(np.float32))/30.

        # For visualization:
        num2show = np.minimum(8, self.config.batch_size*self.config.use_cam_batch)
        # Take half from front & back
        self.show_these = tf.constant(
            np.hstack(
                [np.arange(num2show / 2), self.config.batch_size*self.config.use_cam_batch - np.arange(4) - 1]),
            tf.int32)
        # self.show_swap_these = tf.constant(
        #     np.hstack(
        #         [np.arange(num2show / 2), 2 * self.config.batch_size*self.config.use_cam_batch - np.arange(4) - 1]),
        #     tf.int32)

        # self.is_training = is_training

        # if self.is_training:
        #     self.build_model(self.is_training)
        # else:
        #     self.build_test_model()
        self.build_model()

        # Logging
        self.init_fn = None
        if self.use_pretrained():
            # Make custom init_fn
            print("Fine-tuning from %s" % self.config.pretrained_model_path)

            self.pre_train_saver = tf.train.Saver(self.E_var)
            self.color_pre_train_saver = tf.train.Saver(self.color_Var + self.var_generator)
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

                # self.pre_train_saver.restore(sess, self.config.model_dir)

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
        # self.summary_writer = tf.summary.FileWriter(self.config.model_dir,self.session.graph)
        # self.save_name = self.summary_writer.get_logdir()

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
        self.E_var.append(self.mean_var)
        init_mean = tf.tile(self.mean_var, [self.config.batch_size*self.config.use_cam_batch, 1])
        return init_mean

    def build_model(self):

        if self.config.data_format == 'NCHW':
            self.image_loader=tf.placeholder(dtype=tf.float32,shape=[self.config.batch_size*self.config.use_cam_batch,3,self.config.img_size,self.config.img_size])
        else:
            self.image_loader = tf.placeholder(dtype=tf.float32,
                                          shape=[self.config.batch_size*self.config.use_cam_batch, self.config.img_size, self.config.img_size, 3])
        if self.config.use_2d_joints:
            self.j2d_gt=tf.placeholder(dtype=tf.float32,shape=[self.config.batch_size*self.config.use_cam_batch,self.config.joints_num,2])
        if self.config.use_3d_joints:
            self.j3d_gt = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size*self.config.use_cam_batch, self.config.joints_num, 3])

        self.bg_loader=tf.placeholder(dtype=tf.float32,shape=[self.config.batch_size*self.config.use_cam_batch,self.config.img_size,self.config.img_size,3])
        self.rotation_loader=tf.placeholder(dtype=tf.float32,shape=[self.config.batch_size*self.config.use_cam_batch,3,3])
        image_loader=self.image_loader
        bg_loader=self.bg_loader

        # encode feature
        self.img_feat,self.E_var=Encoder_resnet(image_loader,weight_decay=self.config.e_wd,reuse=False)
        # regress smpl pars
        theta_prev = self.load_mean_param()

        # For visualizations
        self.all_verts = []
        self.all_pred_kps = []
        self.all_pred_cams = []
        self.all_delta_thetas = []
        self.all_theta_prev = []


        if self.config.smpl_prior:
            fake_rotations=[]
            fake_shapes=[]

        # Main IEF loop
        for i in np.arange(self.config.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, theta_prev], 1)

            if i == 0:
                delta_theta, threeD_var = Encoder_fc3_dropout(
                    state,
                    num_output=self.total_params,
                    reuse=False)
                self.E_var.extend(threeD_var)
            else:
                delta_theta, _ = Encoder_fc3_dropout(
                    state, num_output=self.total_params, reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]
            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            verts, Js, pred_Rs = self.human_model(shapes, poses, get_skin=True)
            pred_kp = batch_orth_proj_idrot(
                Js, cams, name='proj2d_stage%d' % i)

            if self.config.smpl_prior:
                # Save pred_rotations for Discriminator
                fake_rotations.append(pred_Rs[:, 1:, :])
                fake_shapes.append(shapes)

            # Save things for visualiations:
            self.all_verts.append(tf.gather(verts, self.show_these))
            self.all_pred_kps.append(tf.gather(pred_kp, self.show_these))
            self.all_pred_cams.append(tf.gather(cams, self.show_these))

            # Finally update to end iteration.
            theta_prev = theta_here

        self.theta_prev=theta_prev
        self.final_poses=poses
        self.final_shapes=shapes
        self.final_cams=cams

        ###########################################rotate###############################################
        # pose = tf.reshape(poses,[self.config.batch_size,self.config.use_cam_batch,-1])
        # pose1 = pose[:,0,:]
        # pose1 = tf.expand_dims(pose1,axis=1)
        # pose234 = pose[:,1:,:]
        # pose234 = tf.reshape(pose234,[-1,72])
        # root_pose234 = pose234[:,:3]
        # root_R = batch_rodrigues(root_pose234)
        # R_all = tf.reshape(self.rotation_loader,[self.config.batch_size,self.config.use_cam_batch,3,3])
        # R_inv = R_all[:,0,:,:]
        # R_inv = tf.expand_dims(R_inv,dim=1)
        # R_inv = tf.tile(R_inv,[1,self.config.use_cam_batch-1,1,1])
        # R_inv = tf.reshape(R_inv,[-1,3,3])
        # rotation_loader_inv = tf.transpose(tf.reshape(R_all[:,1:,:,:],[-1,3,3]),[0,2,1])
        #
        # Rcam2cam = tf.matmul(R_inv,rotation_loader_inv)
        # rotate_r = tf.matmul(Rcam2cam,root_R)
        # self.rotate_r = rotate_r
        # from tf_smpl.batch_lbs import batch_inv_rodrigues2
        # root_pose_rotate,self.theta,self.m,self.ax = batch_inv_rodrigues2(rotate_r)
        # root_pose_rotate = tf.reshape(root_pose_rotate,[self.config.batch_size,self.config.use_cam_batch-1,3])
        # pose234 = tf.reshape(pose234,[self.config.batch_size,self.config.use_cam_batch-1,-1])
        # self.pose_rotate = tf.concat([root_pose_rotate,pose234[:,:,3:]],axis=2)
        #
        # self.pose_rotate = tf.concat([pose1,self.pose_rotate],axis=1)
        # self.pose_rotate = tf.reshape(self.pose_rotate,[-1,72])
        ################################################################################################

        # self.real_rot3=batch_inv_rodrigues(batch_worlds_relate(rotation_loader))
        # self.final_poses=tf.concat([tf.zeros_like(self.pred_poses[:,:3]),self.pred_poses[:,3:]],axis=-1)
        # self.unrotate_poses=self.pred_poses
        # self.unrotate_verts,self.unrotate_Js,self.unrotate_Rs=self.human_model(self.final_shapes,self.unrotate_poses,get_skin=True,name='model_1')
        # self.final_verts=tf.stack([-self.final_verts[:,:,0],self.final_verts[:,:,2],self.final_verts[:,:,1]],axis=-1)
        # self.final_verts=tf.matmul(self.final_verts,tf.transpose(rotation_loader,[0,2,1]))
        # self.unrotate_verts_2d=self.proj_fn(self.unrotate_verts,self.final_cams)
        # self.unrotate_kp=batch_orth_proj_idrot(self.unrotate_Js,self.final_cams,name='proj2d_final')

        if self.config.smpl_prior:
            self.setup_smpl_discriminator(fake_rotations, fake_shapes)
        #get UV smpl image
        self.final_verts, self.final_Js, self.final_Rs = self.human_model(self.final_shapes, self.final_poses,get_skin=True,name='model_2')
        R_all = tf.reshape(self.rotation_loader, [self.config.batch_size, self.config.use_cam_batch, 3, 3])
        R_inv = R_all[:, 0, :, :]
        R_inv = tf.expand_dims(R_inv, dim=1)
        R_inv = tf.tile(R_inv, [1, self.config.use_cam_batch, 1, 1])
        R_inv = tf.reshape(R_inv, [-1, 3, 3])
        rotation_loader_inv = tf.transpose(self.rotation_loader, [0, 2, 1])
        Rcam2cam = tf.matmul(R_inv, rotation_loader_inv)
        self.rotate_verts = tf.matmul(self.final_verts,tf.transpose(Rcam2cam,[0,2,1]))
        self.rotate_Js = tf.matmul(self.final_Js,tf.transpose(Rcam2cam,[0,2,1]))

        self.final_verts_2d=self.proj_fn(self.final_verts,self.final_cams)
        self.final_kp=batch_orth_proj_idrot(self.final_Js,self.final_cams,name='proj2d_final')

        self.rotate_verts_2d = self.proj_fn(self.rotate_verts, self.final_cams)
        self.rotate_kp = batch_orth_proj_idrot(self.rotate_Js, self.final_cams, name='proj2d_rotate')

        self.smpl_uv = dirt.rasterise_batch(
            background=tf.zeros_like(image_loader),
            vertices=tf.concat([self.final_verts_2d,(3.+self.final_verts[:,:,2:3])/(tf.expand_dims(tf.expand_dims(tf.reduce_max(3.+self.final_verts[:,:,2],axis=-1),axis=-1),axis=-1)+1e-10),tf.ones_like(self.final_verts_2d[:,:,0:1])],axis=-1),
            faces=tf.stack([self.human_model.faces] * self.config.batch_size * self.config.use_cam_batch, axis=0),
            vertex_colors=tf.stack([tf.concat([self.human_uv, tf.zeros_like(self.human_uv[:, 0:1])],
                                              axis=-1)] * self.config.batch_size * self.config.use_cam_batch, axis=0),
            height=self.config.img_size,
            width=self.config.img_size,
            channels=3,
            name='renderer')

        self.vertices = tf.concat([self.rotate_verts_2d,(3.+self.rotate_verts[:,:,2:3])/(tf.expand_dims(tf.expand_dims(tf.reduce_max(3.+self.rotate_verts[:,:,2],axis=-1),axis=-1),axis=-1)+1e-10),tf.ones_like(self.rotate_verts_2d[:,:,0:1])],axis=-1)
        self.smpl_uv_rotate=dirt.rasterise_batch(
            background=tf.zeros_like(image_loader),
            vertices= self.vertices,
            faces=tf.stack([self.human_model.faces]*self.config.batch_size*self.config.use_cam_batch,axis=0),
            vertex_colors=tf.stack([tf.concat([self.human_uv,tf.zeros_like(self.human_uv[:,0:1])],axis=-1)]*self.config.batch_size*self.config.use_cam_batch,axis=0),
            height=self.config.img_size,
            width=self.config.img_size,
            channels=3,
            name='renderer_rotate')

        self.smpl_uv=tf.reverse(self.smpl_uv[:,:,:,:2],axis=[1])
        self.smpl_uv_rotate = tf.reverse(self.smpl_uv_rotate[:, :, :, :2], axis=[1])
        #regress color code
        #
        # color_feat,self.color_Var=Color_Encoder_resnet(
        #     tf.concat([self.image_loader,self.smpl_uv],axis=-1),
        #     weight_decay=self.config.e_wd,
        #     reuse=False)

        color_feat, self.color_Var = Color_Encoder_resnet(
            tf.concat([self.image_loader, self.smpl_uv], axis=-1),
            weight_decay=self.config.e_wd,
            reuse=False)

        color_code, color_fc_Var = Encoder_fc3_dropout(color_feat,
                                                       num_output=self.config.color_length,
                                                       reuse=False,
                                                       name='color_stage')
        self.color_Var.extend(color_fc_Var)
        # self.color_Var.extend(color_fc_Var)
        # self.color_Var = color_Var
        self.final_mask,self.final_fore,self.var_generator=InfoGenerator(self.smpl_uv_rotate,color_code,name='generator')
        self.final_img=(1.-self.final_mask)*tf.image.resize_bilinear(bg_loader,[256,256])+(self.final_mask)*self.final_fore
        self.final_img=tf.image.resize_bilinear(self.final_img,[self.config.img_size,self.config.img_size])

        if self.config.img_prior:
            self.setup_image_discriminator(self.final_img)

        #gather reconstruction loss
        groundtruth_img = tf.reshape(image_loader,[self.config.batch_size,self.config.use_cam_batch,self.config.img_size,self.config.img_size,3])[:,0,:,:]
        groundtruth_img = tf.expand_dims(groundtruth_img,dim=1)
        groundtruth_img  =tf.tile(groundtruth_img,[1,self.config.use_cam_batch,1,1,1])
        groundtruth_img = tf.reshape(groundtruth_img,[-1,self.config.img_size,self.config.img_size,3])

        fake_feature_list, self.resnet18_var = self.feature_loss.build_tower(
            tf.image.resize_bilinear(self.move_mean_and_var(self.final_img), [128, 128])
            , reuse=False)
        real_feature_list, _ = self.feature_loss.build_tower(
            tf.image.resize_bilinear(self.move_mean_and_var(groundtruth_img), [128, 128])
            , reuse=True)
        feature_loss_list = [tf.reduce_mean(tf.abs(real_feature_list[i] - fake_feature_list[i])) for i in
                             range(len(fake_feature_list))]
        self.feature_loss = tf.reduce_mean(tf.stack(feature_loss_list, axis=0))
        self.loss_recons=tf.reduce_mean((self.final_img-groundtruth_img)**2)
        self.e_loss = self.config.recons_weight * self.loss_recons \
                      + self.config.e_feature_loss_weight * self.feature_loss
        if self.config.smpl_prior:
            self.e_loss+=self.e_smpl_loss_disc
            self.d_smpl_loss =self.d_smpl_loss_fake+self.d_smpl_loss_real
        if self.config.img_prior:
            self.e_loss += self.config.d_img_loss_weight * self.e_img_loss_disc
            if self.config.discriminator_loss_type == 'wgan':
                self.d_img_loss = self.config.d_img_loss_weight * (
                self.d_img_loss_fake + self.d_img_loss_real + 10.0 * self.gp)
            else:
                self.d_img_loss = self.config.d_img_loss_weight * (self.d_img_loss_fake + self.d_img_loss_real)
        if self.config.use_2d_joints:
            self.loss_2d_joints = self.keypoint_l1_loss(self.j2d_gt,self.final_kp)
            self.j2d_gt_simple = tf.reshape(self.j2d_gt,[self.config.batch_size,self.config.use_cam_batch,-1,2])
            self.j2d_gt_simple = self.j2d_gt_simple[:,0,:,:]
            self.j2d_gt_simple = tf.expand_dims(self.j2d_gt_simple,axis=1)
            self.j2d_gt_simple = tf.tile(self.j2d_gt_simple,[1,4,1,1])
            self.loss_2d_joints += self.keypoint_l1_loss(self.j2d_gt_simple,self.rotate_kp)
            self.e_loss+=self.config.e_j2d_weight*self.loss_2d_joints
        if self.config.use_3d_joints:
            self.loss_3d_joints = self.get_3d_loss(self.j3d_gt,self.final_Js)
            self.j3d_gt_simple = tf.reshape(self.j3d_gt, [self.config.batch_size, self.config.use_cam_batch, -1, 3])
            self.j3d_gt_simple = self.j3d_gt_simple[:, 0, :, :]
            self.j3d_gt_simple = tf.expand_dims(self.j3d_gt_simple, axis=1)
            self.j3d_gt_simple = tf.tile(self.j3d_gt_simple, [1, 4, 1, 1])
            self.loss_3d_joints +=  self.get_3d_loss(self.j3d_gt_simple, self.rotate_Js)
            self.e_loss += self.config.e_j3d_weight *self.loss_3d_joints

        # Don't forget to update batchnorm's moving means.
        print('collecting batch norm moving means!!')
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            self.e_loss = control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.e_loss)
            if self.config.img_prior:
                self.d_img_loss=control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.d_img_loss)
            if self.config.smpl_prior:
                self.d_smpl_loss=control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.d_smpl_loss)

            # Setup optimizer
        print('Setting up optimizer..')
        d_smpl_optimizer = self.optimizer(self.config.d_smpl_lr)
        d_img_optimizer = self.optimizer(self.config.d_img_lr)
        e_optimizer = self.optimizer(self.config.e_lr)

        self.e_opt = e_optimizer.minimize(
            self.e_loss, global_step=self.global_step, var_list=self.color_Var+self.var_generator)
        if self.config.smpl_prior:
            self.d_smpl_opt = d_smpl_optimizer.minimize(self.d_smpl_loss, var_list=self.D_smpl_var)
        if self.config.img_prior:
            self.d_img_opt=d_img_optimizer.minimize(self.d_img_loss,var_list=self.D_img_var)
            if self.config.discriminator_type=='began':
                with tf.control_dependencies([self.e_opt, self.d_img_opt]):
                    self.k_update = tf.assign(
                        self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.setup_summaries()

        print('Done initializing trainer!')

    def align_by_pelvis(self,joints):
        """
        Assumes joints is N x 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        with tf.name_scope("align_by_pelvis", [joints]):
            left_id = 3
            right_id = 2
            pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
            return joints - tf.expand_dims(pelvis, axis=1)

    def visualize_img(self, input,output, gt_kp, pred_kp):
        """
        Overlays gt_kp and pred_kp on img.
        Draws vert with text.
        Renderer is an instance of SMPLRenderer.
        """


        # Draw skeleton

        gt_joint = np.transpose(((gt_kp[:,:, :2] + 1) * 0.5) * self.config.img_size,[0,2,1])
        pred_joint = np.transpose(((pred_kp + 1) * 0.5) * self.config.img_size,[0,2,1])
        input_=[]
        output_=[]
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
        input_=np.stack(input_,axis=0)
        output_=np.stack(output_,axis=0)
        # import ipdb
        # ipdb.set_trace()

        combined = np.concatenate([input_,output_],axis=-2)
        return combined

    def get_3d_loss(self,j3d_gt,j3d):
        pred_joints=j3d[:,:14,:]
        pred_joints=self.align_by_pelvis(pred_joints)
        pred_joints = tf.reshape(pred_joints, [self.config.batch_size*self.config.use_cam_batch, -1])

        gt_joints = tf.reshape(j3d_gt, [self.config.batch_size*self.config.use_cam_batch, 14, 3])
        gt_joints = self.align_by_pelvis(gt_joints)
        gt_joints = tf.reshape(gt_joints, [self.config.batch_size*self.config.use_cam_batch, -1])

        loss_3d_joints=tf.losses.mean_squared_error(
            pred_joints,gt_joints) * 0.5
        return loss_3d_joints

    def keypoint_l1_loss(self,kp_gt, kp_pred, scale=1., name=None):
        """
        computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
        Inputs:
          kp_gt  : N x K x 2
          kp_pred: N x K x 2
        """
        with tf.name_scope(name, "keypoint_l1_loss", [kp_gt, kp_pred]):
            # print(kp_gt.shape,kp_pred.shape)
            kp_gt = tf.reshape(kp_gt, (-1, 2))
            kp_pred = tf.reshape(kp_pred[:,:14,:], (-1, 2))

            vis = tf.expand_dims(tf.cast(tf.ones_like(kp_gt[:, 1]), tf.float32), 1)
            loss_2d_joints = tf.losses.absolute_difference(kp_gt[:, :2], kp_pred, weights=vis)
            return loss_2d_joints


    def draw_results(self,result):

        from StringIO import StringIO
        import matplotlib.pyplot as plt
        show_total_image = result['show_total_image']
        if self.config.use_2d_joints:

            show_kps_image=self.visualize_img(result["show_input"], result["show_out"],
                               result["show_kps_gt"], result["show_kps"])
            show_kps_image=np.stack([show_kps_image[:,:,:,2],show_kps_image[:,:,:,1],
                                     show_kps_image[:,:,:,0]],axis=-1)
        img_summaries=[]

        for i in range(show_total_image.shape[0]):
            sio = StringIO()
            if self.config.use_2d_joints:
                new=(np.concatenate([show_total_image[i], show_kps_image[i]], axis=0)).astype(np.uint8)
            else:
                new=show_total_image[i].astype(np.uint8)
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



    def setup_smpl_discriminator(self, fake_rotations, fake_shapes):
        # Compute the rotation matrices of "rea" pose.
        # These guys are in 24 x 3.
        self.pose_loader=tf.placeholder(shape=[self.config.batch_size*self.config.use_cam_batch,24*3])
        real_rotations = batch_rodrigues(tf.reshape(self.pose_loader, [-1, 3]))
        real_rotations = tf.reshape(real_rotations, [-1, 24, 9])
        # Ignoring global rotation. N x 23*9
        # The # of real rotation is B*num_stage so it's balanced.
        real_rotations = real_rotations[:, 1:, :]
        all_fake_rotations = tf.reshape(
            tf.concat(fake_rotations, 0),
            [self.config.batch_size*self.config.use_cam_batch * self.config.num_stage, -1, 9])
        comb_rotations = tf.concat(
            [real_rotations, all_fake_rotations], 0, name="combined_pose")

        self.shape_Loader=tf.placeholder(shape=[self.config.batch_size*self.config.use_cam_batch,10])
        comb_rotations = tf.expand_dims(comb_rotations, 2)
        all_fake_shapes = tf.concat(fake_shapes, 0)
        comb_shapes = tf.concat(
            [self.shape_loader, all_fake_shapes], 0, name="combined_shape")

        disc_input = {
            'weight_decay': self.config.d_wd,
            'shapes': comb_shapes,
            'poses': comb_rotations
        }

        self.d_out, self.D_smpl_var = Discriminator_separable_rotations(
            **disc_input)

        self.d_out_real, self.d_out_fake = tf.split(self.d_out, 2)
        # Compute losses:
        with tf.name_scope("comp_d_smpl_loss"):
            self.d_smpl_loss_real = tf.reduce_mean(
                tf.reduce_sum((self.d_out_real - 1)**2, axis=1))
            self.d_smpl_loss_fake = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake)**2, axis=1))
            # Encoder loss
            self.e_smpl_loss_disc = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake - 1)**2, axis=1))

    def setup_image_discriminator(self, fake_images):
        self.real_images = tf.placeholder(dtype=tf.float32,
                                          shape=[self.config.batch_size, self.config.img_size, self.config.img_size,
                                                 3])
        if self.config.discriminator_type == 'norm' or self.config.discriminator_type == 'star':
            if self.config.discriminator_type == 'norm':
                self.discriminator = NLayerDiscriminator
            else:
                self.discriminator = StarDiscriminator
            self.real_score, self.D_img_var = self.discriminator(name='img_dis', x=self.real_images, reuse=False)
            self.fake_score, _ = self.discriminator(name='img_dis', x=fake_images, reuse=True)
            # print(self.real_score.shape)
            # import ipdb
            # ipdb.set_trace()
            # compute losses:

            with tf.name_scope("comp_d_img_loss"):
                if self.config.discriminator_loss_type == 'norm':
                    self.d_img_loss_real = tf.reduce_mean(
                        tf.reduce_mean((self.real_score - 1) ** 2, axis=1))
                    self.d_img_loss_fake = tf.reduce_mean(
                        tf.reduce_mean((self.fake_score) ** 2, axis=1))
                    # Encoder loss
                    self.e_img_loss_disc = tf.reduce_mean(
                        tf.reduce_mean((self.fake_score - 1) ** 2, axis=1))
                elif self.config.discriminator_loss_type == 'wgan':
                    def gradient_penalty(real, fake):
                        def interpolate(a, b):
                            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
                            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                            inter = a + alpha * (b - a)
                            inter.set_shape(a.get_shape().as_list())
                            return inter

                        x = interpolate(real, fake)
                        pred, _ = self.discriminator(name='img_dis', x=x, reuse=True)
                        gradients = tf.gradients(pred, x)[0]
                        slopes = tf.sqrt(
                            tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
                        gp = tf.reduce_mean((slopes - 1.) ** 2)
                        return gp

                    self.gp = gradient_penalty(self.real_images, fake_images)

                    self.d_img_loss_real = tf.reduce_mean(
                        -self.real_score)
                    self.d_img_loss_fake = tf.reduce_mean(
                        self.fake_score)
                    # Encoder loss
                    self.e_img_loss_disc = tf.reduce_mean(
                        -self.fake_score)

        elif self.config.discriminator_type == 'began':
            self.k_t = tf.Variable(0., trainable=False, name='k_t')
            self.real_out, self.real_z, self.D_img_var = DiscriminatorBEGAN(name='img_dis', x=self.real_images,
                                                                            reuse=False,
                                                                            repeat_num=int(
                                                                                np.log(self.config.img_size)) - 2)

            self.fake_out, self.fake_z, _ = DiscriminatorBEGAN(name='img_dis', x=fake_images,
                                                               reuse=True,
                                                               repeat_num=int(
                                                                   np.log(self.config.img_size)) - 2)

            self.d_img_loss_real = tf.reduce_mean(tf.abs(self.real_out - self.real_images))
            self.d_img_loss_fake = tf.reduce_mean(tf.abs(self.fake_out - fake_images))
            self.d_img_loss = self.d_img_loss_real - self.k_t * self.d_img_loss_fake
            self.e_img_loss_disc = self.d_img_loss_fake
            self.gamma = 0.5
            self.lambda_k = 0.001
            self.balance = self.gamma * self.d_img_loss_real - self.e_img_loss_disc
            self.measure = self.d_img_loss_real + tf.abs(self.balance)

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
        always_report=[
            tf.summary.scalar("loss/loss_recons",self.loss_recons),
            tf.summary.scalar('loss/loss_feature', self.feature_loss),
            tf.summary.scalar('loss/e_loss',self.e_loss)

        ]
        if self.config.smpl_prior:
            always_report.extend([
                tf.summary.scalar("loss/d_smpl_loss", self.d_smpl_loss),
                tf.summary.scalar("loss/d_smpl_loss_real", self.d_smpl_loss_real),
                tf.summary.scalar("loss/d_smpl_loss_fake", self.d_smpl_loss_fake),
                tf.summary.scalar("loss/e_smpl_loss_disc", self.e_smpl_loss_disc),
            ])

        if self.config.img_prior:
            always_report.extend([
                tf.summary.scalar("loss/d_img_loss", self.d_img_loss),
                tf.summary.scalar("loss/d_img_loss_real", self.d_img_loss_real),
                tf.summary.scalar("loss/d_img_loss_fake", self.d_img_loss_fake),
                tf.summary.scalar("loss/e_img_loss_disc", self.e_img_loss_disc),
            ])
            if self.config.discriminator_type=='began':
                always_report.extend([
                    tf.summary.scalar("loss/k_t", self.k_t),
                    tf.summary.scalar("loss/balance", self.balance),

                ])
        if self.config.use_2d_joints:
            always_report.extend([
                tf.summary.scalar("loss/loss_2d_joints",self.loss_2d_joints),
            ])
        if self.config.use_3d_joints:
            always_report.extend([
                tf.summary.scalar("loss/loss_3d_joints", self.loss_3d_joints),
            ])

        self.summary_op_always=tf.summary.merge(always_report)

        image_report=[]
        input_image=(tf.gather(self.image_loader,self.show_these)+1.)*255./2.
        bg_image=(tf.gather(self.bg_loader,self.show_these)+1.)*255./2.
        u_image=(tf.stack([tf.gather(self.smpl_uv,self.show_these)[:,:,:,0]]*3,-1))*255.
        v_image = (tf.stack([tf.gather(self.smpl_uv,self.show_these)[:,:,:,1]]*3,-1))*255.
        mask=(tf.concat([tf.gather(self.final_mask,self.show_these)]*3,-1))*255.
        fore=(tf.gather(self.final_fore,self.show_these)+1.)*255./2.
        final=(tf.gather(self.final_img,self.show_these)+1.)*255./2.
        dis=tf.abs(final-input_image)
        total_image=tf.concat([
            tf.concat([input_image,u_image,tf.image.resize_bilinear(mask,[self.config.img_size,self.config.img_size]),bg_image],axis=1),
            tf.concat([final,v_image,tf.image.resize_bilinear(fore,[self.config.img_size,self.config.img_size]),dis],axis=1),
            ],axis=2)

        self.show_total_image=tf.stack([total_image[:,:,:,2],
                              total_image[:, :, :, 1],
                              total_image[:, :, :, 0]],axis=-1)

        self.show_input=input_image
        self.show_out=final
        if self.config.use_2d_joints:
            self.show_kp_gt=tf.gather(self.j2d_gt,self.show_these)
            self.show_kp=tf.gather(self.final_kp[:,:14,:],self.show_these)
            # self.show_kp = tf.gather(self.final_kp[:, :14, :], self.show_these)




    def train(self):
        # For rendering!
        step = 0
        self.session.run(tf.global_variables_initializer())
        if self.init_fn is not None:
            self.init_fn(self.session)

        while True:
            fetch_dict = {
                "summary": self.summary_op_always,
                "step": self.global_step,
                # The meat
                "e_opt": self.e_opt,
                "e_loss": self.e_loss,
                "loss_recons": self.loss_recons,
                "feature_loss":self.feature_loss,
            }
            if self.config.smpl_prior:
                fetch_dict.update({
                    # For D:
                    "d_smpl_opt": self.d_smpl_opt,
                    "d_smpl_loss": self.d_smpl_loss,
                    "loss_smpl_disc": self.e_smpl_loss_disc,
                })
            if self.config.img_prior:
                fetch_dict.update({
                    # For D:
                    "d_img_opt": self.d_img_opt,
                    "d_img_loss": self.d_img_loss,
                    "loss_img_disc": self.e_img_loss_disc,
                })

            if step % self.config.log_img_step == 0:
                fetch_dict.update({
                    "show_total_image": self.show_total_image,

                })
                if self.config.use_2d_joints:
                    fetch_dict.update({
                        "show_input": self.show_input,
                        "show_out": self.show_out,
                        "show_kps_gt": self.show_kp_gt,
                        "show_kps": self.show_kp,

                    })
            if self.config.use_2d_joints or self.config.use_3d_joints:
                img, bg, R, ID,D2,D3=next(self.data_iter)
                D2 = np.reshape(D2,[self.config.batch_size*self.config.use_cam_batch,-1,2])
                D3 = np.reshape(D3,[self.config.batch_size*self.config.use_cam_batch,-1,3])
            else:
                img, bg, R, ID= next(self.data_iter)
            img = np.reshape(img,[self.config.batch_size*self.config.use_cam_batch,self.config.img_size,self.config.img_size,3])
            bg = bg[:,0,:,:,:]
            bg = np.expand_dims(bg,axis=1)
            bg = np.tile(bg,[1,self.config.use_cam_batch,1,1,1])
            bg = np.reshape(bg,[self.config.batch_size*self.config.use_cam_batch,self.config.img_size,self.config.img_size,3])
            R = np.reshape(R,[self.config.batch_size*self.config.use_cam_batch,3,3])
            # ID = tf.reshape(ID,[self.config.batch_size*self.config.use_cam_batch,4])
            # feed_dict={
            #     self.image_loader:img[:,0,:,:,:],
            #     self.bg_loader:bg[:,0,:,:,:],
            #     self.rotation_loader:R[:,0,:,:],
            # }
            feed_dict = {
                self.image_loader:img,
                self.bg_loader:bg,
                self.rotation_loader:R,
            }
            if self.config.use_2d_joints:
                feed_dict.update({
                    self.j2d_gt:D2,
                })
            if self.config.use_3d_joints:
                feed_dict.update({
                    self.j3d_gt:D3,
                })

            if self.config.img_prior:
                if self.real_data_iter is None:
                    print('No real images')
                    import ipdb
                    ipdb.set_trace()
                real_img, real_bg, real_R, real_ID = next(self.real_data_iter)
                real_img = tf.reshape(real_img,[self.config.batch_size*self.config.use_cam_batch])
                feed_dict.update(
                    {self.real_images:real_img}
                )
            # if 1:
            #     fetch_dict.update({
            #         "j3d_gt":self.j3d_gt[:,:14,:],
            #         "j3d":self.final_Js[:,:14,:],
            #         "input":self.image_loader,
            #     })


            t0 = time()
            result = self.session.run(fetch_dict,feed_dict=feed_dict)

            t1 = time()


            # if 1:
            #     import cv2
            #     cv2.imwrite('test.png',(result['input'][0]+1.)*255./2.)
            #     print(result['j3d'].shape,result['j3d_gt'].shape)
            #     # import ipdb
            #     # ipdb.set_trace()
            #
            #     plot_j3d(result['j3d'][0],result['j3d_gt'][0])

            self.summary_writer.add_summary(
                result['summary'], global_step=result['step'])

            e_loss = result['e_loss']
            step = result['step']

            epoch = float(step) / self.num_itr_per_epoch

            print("itr %d/(epoch %.1f): time %g, Enc_loss: %.4f" %
                       (step, epoch, t1 - t0, e_loss))

            if step%10000==0:
                self.saver.save(self.session,os.path.join(self.save_name,'model.ckpt'),global_step=result['step'])

            if step % self.config.log_img_step == 0:
                self.draw_results(result)

            self.summary_writer.flush()
            if epoch > self.config.epoch:
                break

            step += 1

        print('Finish training on %s' % self.config.model_dir)






