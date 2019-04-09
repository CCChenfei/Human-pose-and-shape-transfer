import numpy as np
import cv2
import os
import random

import h5py
import cPickle as pickle
import tqdm

class DataLoader():
    def __init__(self, config,randomize=False):
        self.img_dir = config.data_dir
        self.useCamBatches = config.use_cam_batch
        # self.input_types = input_types
        # self.label_types = label_types
        self.randomrize = randomize
        self.img_res=config.img_size
        self.batch_size=config.batch_size

        h5_label_file = h5py.File(self.img_dir+'/labels.h5','r')
        print(self.img_dir+'/labels.h5 loading done')
        self.label_dict = {key:np.array(value) for key,value in h5_label_file.items()}

        all_keys_name = self.img_dir + '/all_keys2.pickl'
        sequence_keys_name =self.img_dir + '/sequence_keys2.pickl'
        camsets_name = self.img_dir + '/camsets2.pickl'
        subject_keys_name = self.img_dir + '/subject_keys2.pickl'
        print('Done loading h5 label file')
        if os.path.exists(sequence_keys_name):
            print('Loading sequence-subject-cam association from pickle files {}.'.format(sequence_keys_name))
            self.all_keys = pickle.load(open(all_keys_name, "rb"))
            self.sequence_keys = pickle.load(open(sequence_keys_name, "rb"))
            self.camsets = pickle.load(open(camsets_name, "rb"))
            self.subject_keys = pickle.load(open(subject_keys_name, "rb"))
            print('Done loading sequence association.')

        else:
            print('Establishing sequence association. Available labels:', list(h5_label_file.keys()))
            all_keys = set()
            camsets = {}
            sequence_keys = {}
            subject_keys = {}
            data_length = len(h5_label_file['frame'])
            with tqdm.tqdm(total=data_length) as pbar:
                for index in range(data_length):
                    pbar.update(1)
                    sub_i = int(h5_label_file['subj'][index].item())
                    cam_i = int(h5_label_file['cam'][index].item())
                    seq_i = int(h5_label_file['seq'][index].item())
                    frame_i = int(h5_label_file['frame'][index].item())

                    key = (sub_i, seq_i, frame_i)
                    if key not in camsets:
                        camsets[key] = {}
                    camsets[key][cam_i] = index

                    # only add if accumulated enough cameras
                    if len(camsets[key]) >= self.useCamBatches:
                        all_keys.add(key)

                        if seq_i not in sequence_keys:
                            sequence_keys[seq_i] = set()
                        sequence_keys[seq_i].add(key)

                        if sub_i not in subject_keys:
                            subject_keys[sub_i] = set()
                        subject_keys[sub_i].add(key)

            self.all_keys = list(all_keys)
            self.camsets = camsets
            self.sequence_keys = {seq: list(keyset) for seq, keyset in sequence_keys.items()}
            self.subject_keys = {sub: list(keyset) for sub, keyset in subject_keys.items()}
            pickle.dump(self.all_keys, open(all_keys_name, "wb"))
            pickle.dump(self.sequence_keys, open(sequence_keys_name, "wb"))
            pickle.dump(self.camsets, open(camsets_name, "wb"))
            pickle.dump(self.subject_keys, open(subject_keys_name, "wb"))
            print("Done initialization")

    def generator(self, normalize=True,cam_shuffle=False,usePoseLabel=True,useLsp=True,shuffle_app=False,finetune=False):
        batch_size=self.batch_size
        img_res=self.img_res


        while True:
            img = np.zeros((batch_size, self.useCamBatches,img_res, img_res, 3), dtype=np.float32)
            bg = np.zeros((batch_size, self.useCamBatches,img_res, img_res,3), dtype=np.float32)
            R = np.zeros((batch_size,self.useCamBatches,3,3),dtype=np.float32)
            ID = np.zeros((batch_size,self.useCamBatches,4),dtype=np.int32)
            if usePoseLabel:
                D2 = np.zeros((batch_size,self.useCamBatches,2,17),dtype=np.float32)
                D3 = np.zeros((batch_size,self.useCamBatches,3,17),dtype=np.float32)
            if self.randomrize:
                random.shuffle(self.all_keys)
            if shuffle_app:
                for i in range(batch_size//2):
                    if finetune:
                        action_list = self.sequence_keys[2]
                        key = random.choice(action_list)
                        while key[0]!= 9:
                            key = random.choice(action_list)
                    else:
                        key = random.choice(self.all_keys)
                    subi = key[0]
                    potential_keys = self.subject_keys[subi]
                    key_other = potential_keys[np.random.randint(len(potential_keys))]

                    camBatches = list(range(1, self.useCamBatches + 1))
                    if cam_shuffle:
                        random.shuffle(camBatches)
                    for cami, cam in enumerate(camBatches):
                        img[2*i][cami] = self.loadImage(key, cam, 'img_crop', img_res, normalize)
                        img[2*i+1][cami] = self.loadImage(key_other, cam, 'img_crop', img_res, normalize)
                        bg[2*i][cami] = self.loadImage(key, cam, 'bg_crop', img_res, normalize)
                        bg[2*i+1][cami] = self.loadImage(key_other, cam, 'bg_crop', img_res, normalize)
                        index = self.camsets[key][cam]
                        index_other = self.camsets[key_other][cam]
                        R[2*i][cami] = np.transpose(self.label_dict['extrinsic_rot'], [2, 1, 0])[index]
                        R[2*i+1][cami] = np.transpose(self.label_dict['extrinsic_rot'], [2, 1, 0])[index_other]
                        id = list(key)
                        id.append(cam)
                        id_other = list(key_other)
                        id_other.append(cam)
                        ID[2*i][cami] = id
                        ID[2 * i+1][cami] = id_other
                        if usePoseLabel:
                            D2[2*i][cami] = np.transpose(self.label_dict['2D'], [2, 1, 0])[index]
                            D2[2*i+1][cami] = np.transpose(self.label_dict['2D'], [2, 1, 0])[index_other]
                            D3cam = np.transpose(self.label_dict['3D'], [2, 1, 0])[index]
                            R_inv = np.transpose(self.label_dict['extrinsic_rot_inv'], [2, 1, 0])[index]
                            D3[2*i][cami] = np.matmul(R_inv, D3cam)
                            D3cam = np.transpose(self.label_dict['3D'], [2, 1, 0])[index_other]
                            R_inv = np.transpose(self.label_dict['extrinsic_rot_inv'], [2, 1, 0])[index_other]
                            D3[2*i+1][cami] = np.matmul(R_inv, D3cam)

            else:
                for i in range(batch_size):

                    key = random.choice(self.all_keys)
                    camBatches = list(range(1,self.useCamBatches+1))
                    if cam_shuffle:
                        random.shuffle(camBatches)
                    for cami,cam in enumerate(camBatches):
                        img[i][cami] = self.loadImage(key,cam,'img_crop',img_res,normalize)
                        bg[i][cami] = self.loadImage(key,cam,'bg_crop',img_res,normalize)
                        index = self.camsets[key][cam]
                        R[i][cami] = np.transpose(self.label_dict['extrinsic_rot'],[2,1,0])[index]
                        id = list(key)
                        id.append(cam)
                        ID[i][cami] = id
                        if usePoseLabel:
                            D2[i][cami] = np.transpose(self.label_dict['2D'],[2,1,0])[index]
                            D3cam = np.transpose(self.label_dict['3D'],[2,1,0])[index]
                            R_inv = np.transpose(self.label_dict['extrinsic_rot_inv'],[2,1,0])[index]
                            D3[i][cami] = np.matmul(R_inv,D3cam)

            if usePoseLabel:
                if usePoseLabel:
                    if useLsp:
                        yield img, bg, R, ID, self.normalize_2dkp(np.transpose(self.h36m_to_lsp(D2), [0, 1, 3, 2])), \
                              self.h36m3d_to_lsp(np.transpose(self.h36m_to_lsp(D3), [0, 1, 3, 2]))
                    else:
                        yield img, bg, R, ID, self.normalize_2dkp(np.transpose(D2, [0, 1, 3, 2])), np.transpose(D3,[0, 1, 3, 2])
            else:
                yield img,bg,R,ID



    def h36m3d_to_lsp(self, j3d):
        # import scipy.io as sio
        # j3d=sio.loadmat('/home/gzp/Downloads/matlab.mat')
        # j3d=j3d['jointsW']
        # print j3d.shape
        return np.stack([-j3d[:, :, :, 0], -j3d[:, :, :, 2], -j3d[:, :, :, 1]], axis=-1)

    def normalize_2dkp(self, kp2d):
        kp2d = (kp2d / 256. - 0.5) * 2.
        return kp2d

    def h36m_to_lsp(self, joints):
        _COMMON_JOINT_IDS = np.array([
            3,  # R ankle
            2,  # R knee
            1,  # R hip
            4,  # L hip
            5,  # L knee
            6,  # L ankle
            16,  # R Wrist
            15,  # R Elbow
            14,  # R shoulder
            11,  # L shoulder
            12,  # L Elbow
            13,  # L Wrist
            8,  # Neck top
            10,  # Head top
        ])
        return joints[:, :, :, _COMMON_JOINT_IDS]

    def loadImage(self,key,cam,type,img_res,normalize):
        img_name = self.img_dir+'/s'+str(key[0])+'/seq'+str(key[1])+'/cam'+str(cam)+'/'+type+str(key[2])+'.jpg'
        img = cv2.imread(img_name)
        img = cv2.resize(img,(img_res,img_res))
        img = img/255.
        if normalize:
            # img_mean = (0.485, 0.456, 0.406)
            # img_std = (0.229, 0.224, 0.225)
            img_mean=(0.5,0.5,0.5)
            img_std=(0.5,0.5,0.5)
            img = (img - img_mean)/img_std
        return img


if __name__ == '__main__':
    dataloader = DataLoader(img_dir='/home/chenf/Documents/pose_estimation/data/H36M-Multiview/train',useCamBatches=4,randomize=False)
    gen_iter = dataloader.generator(batch_size=4,img_res=256,normalize=True,cam_shuffle=True)
    for i in range(4):
        img,bg,R,ID = next(gen_iter)
        print("j")