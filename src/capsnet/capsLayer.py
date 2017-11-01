import numpy as np
import tensorflow as tf

class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, conv_num_outputs, fcc_num_outputs, conv_vec_len, fcc_vec_len, caps_neurons, batch_size):
        self.conv_num_outputs = conv_num_outputs
        self.fcc_num_outputs = fcc_num_outputs
        self.conv_vec_len = conv_vec_len
        self.fcc_vec_len = fcc_vec_len
        self.caps_neurons = caps_neurons
        self.batch_size = batch_size
        self.iter_routing = 3
        self.stddev = 0.01

    def conv_layer(self, input, kernel_size=None, stride=None):
        capsules = []
        for i in range(self.conv_vec_len):
            with tf.variable_scope('ConvUnit_' + str(i)):
                caps_i = tf.contrib.layers.conv2d(input, self.conv_num_outputs,
                                                  kernel_size, stride,
                                                  padding="VALID")
                caps_i = tf.reshape(caps_i, shape=(self.batch_size, -1, 1, 1))
                capsules.append(caps_i)


        capsules = tf.concat(capsules, axis=2)
        capsules = self.squash(capsules)

        return(capsules)

    def fcc_layer(self, input):
        input = tf.reshape(input, shape=(self.batch_size, self.caps_neurons, 1, self.conv_vec_len, 1))

        with tf.variable_scope('routing'):
            # b_IJ: [1, 1, num_caps_l, num_caps_l_plus_1, 1]
            b_IJ = tf.zeros(shape=[1, self.caps_neurons, self.fcc_num_outputs, 1, 1], dtype=np.float32)
            capsules = self.routing(input, b_IJ)
            capsules = tf.squeeze(capsules, axis=1)

        return(capsules)



    def routing(self, input, b_IJ):
        ''' The routing algorithm.

        Args:
            input: A Tensor with [batch_size, 1, num_caps_l=1152, length(u_i)=8, 1]
                   shape, num_caps_l meaning the number of capsule in the layer l.
        Returns:
            A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
            representing the vector output `v_j` in the layer l+1
        Notes:
            u_i represents the vector output of capsule i in the layer l, and
            v_j the vector output of capsule j in the layer l+1.
         '''

        # W: [num_caps_j, num_caps_i, len_u_i, len_v_j]
        W = tf.get_variable('Weight', shape=(1, self.caps_neurons, self.fcc_num_outputs, self.conv_vec_len, self.fcc_vec_len), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=self.stddev))

        # Eq.2, calc u_hat
        # do tiling for input and W before matmul
        # input => [batch_size, 1152, 10, 8, 1]
        # W => [batch_size, 1152, 10, 8, 16]
        input = tf.tile(input, [1, 1, self.fcc_num_outputs, 1, 1])
        W = tf.tile(W, [self.batch_size, 1, 1, 1, 1])

        # in last 2 dims:
        # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
        u_hat = tf.matmul(W, input, transpose_a=True)

        # line 3,for r iterations do
        for r_iter in range(self.iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
                # => [1, 1, 1152, 10, 1]
                c_IJ = tf.nn.softmax(b_IJ, dim=3)
                c_IJ = tf.tile(c_IJ, [self.batch_size, 1, 1, 1, 1])


                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last tow dim
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)

                # line 6:
                # squash using Eq.1,
                v_J = self.squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 10, 1152, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, self.caps_neurons, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat, v_J_tiled, transpose_a=True)
                print("routing.u_produce_v.shape", u_produce_v.get_shape())
                b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)

        return(v_J)


    def squash(self, vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A 5-D tensor with shape [batch_size, 1, num_caps, vec_len, 1],
        Returns:
            A 5-D tensor with the same shape as vector but squashed in 4rd and 5th dimensions.
        '''
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm)
        vec_squashed = scalar_factor * vector  # element-wise
        return(vec_squashed)
