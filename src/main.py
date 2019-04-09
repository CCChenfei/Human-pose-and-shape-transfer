import tensorflow as tf

from config import get_config, prepare_dirs, save_config
from dataloader.dataloader import DataLoader
from trainer_v2 import Fusion4DTrainer

def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        data_iter = data_loader.generator(usePoseLabel=(config.use_2d_joints or
                                                        config.use_3d_joints),
                                          shuffle_app=config.use_swap_uv)

        if config.img_prior:
            real_data_loader=DataLoader(config)
            real_data_iter=real_data_loader.generator(usePoseLabel=False)
        else:
            real_data_iter=None


    trainer = Fusion4DTrainer(config, data_iter,real_data_iter)
    save_config(config)
    trainer.train()


if __name__ == '__main__':
    config = get_config()
    main(config)