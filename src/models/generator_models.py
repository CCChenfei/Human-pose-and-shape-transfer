
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer

def InfoGenerator(x,
                  c,
                is_training=True,
                reuse=False,
                name="info_generator"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal:
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    batch_size=x.shape[0].value
    x_nc=x.shape[-1].value

    def UpBlock(input,is_training,name=None,reuse=False):
        height,width,nc=input.shape[1],input.shape[2],input.shape[3]
        with tf.variable_scope(name, reuse=reuse) as scope:
            input=tf.image.resize_bilinear(input,[height*2,width*2])
            input=slim.conv2d(input,nc*2,kernel_size=3,stride=1)
            input=slim.batch_norm(input,is_training=is_training)
            return input[:,:,:,:nc]*tf.nn.sigmoid(input[:,:,:,nc:])

    def ResBlock(input,is_training,name=None,reuse=False):
        height,width,nc=input.shape[1].value,input.shape[2].value,input.shape[3].value

        with tf.variable_scope(name, reuse=reuse) as scope:
            input_2=slim.conv2d(input,2*nc,kernel_size=3)
            input_2=slim.batch_norm(input_2,is_training=is_training)
            input_2=input_2[:,:,:,:nc]*tf.nn.sigmoid(input_2[:,:,:,nc:])
            input_2 = slim.conv2d(input_2, nc,kernel_size=3)
            input_2 = slim.batch_norm(input_2,is_training=is_training)
            return input+input_2



    if reuse:
        print('Reuse is on!')
    with tf.variable_scope(name, reuse=reuse) as scope:
        #GLU
        c=slim.fully_connected(c,num_outputs=64*4*4*2)
        c=slim.batch_norm(c,is_training=is_training)
        c=c[:,:64*4*4]*tf.nn.sigmoid(c[:,64*4*4:])
        c=tf.reshape(c,[batch_size,4,4,64])
        #size 8x8
        c=UpBlock(c,is_training=is_training,name='UpBlock_0')
        c = UpBlock(c, is_training=is_training, name='UpBlock_1')
        c = UpBlock(c, is_training=is_training, name='UpBlock_2')
        c = UpBlock(c, is_training=is_training, name='UpBlock_3')
        #size 64x64

        #resize x
        x=tf.image.resize_bilinear(x,[256,256])
        x=slim.conv2d(x,x_nc*2,kernel_size=3,stride=2)
        x=slim.batch_norm(x,is_training=is_training)
        x = slim.conv2d(x, x_nc * 4, kernel_size=3, stride=2)
        x = slim.batch_norm(x,is_training=is_training)

        x_and_c=tf.concat([x,c],-1)
        for i in range(3):
            x_and_c=ResBlock(x_and_c,is_training=is_training,name='resblock_%d'%i)

        out=slim.conv2d_transpose(x_and_c,64/2,3,2)
        out=slim.batch_norm(out,is_training=is_training)
        out = slim.conv2d_transpose(out, 64 / 4, 3, 2)
        out = slim.batch_norm(out, is_training=is_training)
        out = slim.conv2d_transpose(out,64/8,3,2)
        out = slim.batch_norm(out,is_training = is_training)

        mask_out=slim.conv2d(out,1,3,activation_fn=slim.nn.sigmoid)
        fore_out=slim.conv2d(out,3,3,activation_fn=slim.nn.tanh)




    variables = tf.contrib.framework.get_variables(scope)
    return mask_out,fore_out, variables