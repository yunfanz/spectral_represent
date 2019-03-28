import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.layers import conv2d, conv2d_transpose, max_pool2d
from triplet_loss import *
class Autoencoder(object):

    def __init__(self, sess=None, input_tensor=None, input_dim=16*128, h=16, w=128, alpha_r=3., 
                 learning_rate=1e-4, batch_size=128, n_z=16, loss_type='CE', use_conv=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.h = h
        self.w = w
        self.input_dim = input_dim
        tf.reset_default_graph()
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        
        self.build(input_tensor, loss_type=loss_type, use_conv=use_conv, alpha_r=alpha_r)
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self, input_tensor=None, loss_type='CE', use_conv=False, sphere_lat=True, crop=-1, alpha_r=3):
        self.input = tf.placeholder(
            dtype=tf.float32, shape=[None, self.h, self.w, 1], name="input_")
        self.label = tf.placeholder(dtype=tf.int32, shape=[None,], name="label_")

        print(self.input.name)
        if crop > 0:
            crops = [tf.random_crop(self.input, size=(self.batch_size, 16,72,1), 
                                seed=None, name="crop_{0:d}".format(ci)) for ci in range(crop)]
            self.input_ = tf.concat(crops, axis=0, name="input_crop")
            self.x = tf.layers.flatten(self.input_, name='x')
            self.input_dim = 16*72
        else:
            
            self.x = tf.layers.flatten(self.input, name='x')
        # Encode
        # x -> z_mean, z_sigma -> z
        for var in tf.global_variables():
            print(var.name, var.get_shape())
        if use_conv:
            if crop > 0:
                f1 = conv2d(self.input_, num_outputs=32, kernel_size=[3,3])
            else:
                f1 = conv2d(self.input, num_outputs=32, kernel_size=[3,3])
            m1 = max_pool2d(f1, kernel_size=[1,2], stride=[1,2])
            f2 = conv2d(m1, num_outputs=64, kernel_size=[3,3])
            m2 = max_pool2d(f2, kernel_size=2, stride=2)
            f3 = conv2d(m2, num_outputs=128, kernel_size=[3,3])
            m3 = max_pool2d(f3, kernel_size=[1,2], stride=[1,2])
            f4 = conv2d(m3, num_outputs=256, kernel_size=[3,3])
            m4 = max_pool2d(f4, kernel_size=2, stride=2)
            f5 = conv2d(m4, num_outputs=128, kernel_size=[3,3])
            m5 = max_pool2d(f5, kernel_size=2, stride=2)
            f6 = conv2d(m5, num_outputs=256, kernel_size=[3,3])
            #m6 = max_pool2d(f6, kernel_size=2, stride=2)
            #f7 = conv2d(m5, num_outputs=256, kernel_size=[3,3])
            ff = tf.reduce_mean(f6, axis=(1,2))
        else:
            f1 = fc(self.x, 512, scope='enc_fc1', activation_fn=tf.nn.relu)
            f2 = fc(f1, 256, scope='enc_fc2', activation_fn=tf.nn.relu)
            ff = fc(f2, 128, scope='enc_fc3', activation_fn=tf.nn.relu)
        z = fc(ff, self.n_z, scope='enc_fc4', activation_fn=None)
        if sphere_lat:
            norm = tf.sqrt(tf.reduce_sum(z*z,1, keepdims=True))
            self.z = tf.div(z, norm, name='enc_norm')
        else:
            self.z = z
        print(self.z.name)
        if crop > 0:
            z1 = tf.reshape(self.z, (1, crop, self.batch_size, -1))
            z2 = tf.reshape(self.z, (crop, 1, self.batch_size, -1))
            zloss = tf.reduce_mean(1-tf.reduce_sum(z1*z2, axis=-1), name='zloss') * 0.1
        else:
            zloss, pos_frac = batch_all_triplet_loss(self.label, self.z, margin=0.1, squared=False) 
            #zloss = zloss * 0.
            #tf.constant(0, dtype=tf.float32)
            
        # Decode
        # z -> x_hat
        g1 = fc(self.z, 128, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 256, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.relu)
        self.x_hat = fc(g3, self.input_dim, scope='dec_fc4', 
                        activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        if loss_type == 'CE':
            self.recon_loss = -tf.reduce_sum(
                self.x * tf.log(epsilon+self.x_hat) + 
                (1-self.x) * tf.log(epsilon+1-self.x_hat), 
                axis=1
            )  * alpha_r
        elif loss_type == 'l2':
            self.recon_loss = tf.sqrt(tf.reduce_mean(
                tf.square(self.x -self.x_hat),
                axis=1
            )) * alpha_r
        elif loss_type == 'l1':
            self.recon_loss = tf.reduce_mean(
                tf.abs(self.x -self.x_hat),
                axis=1
            ) * alpha_r
#         self.target_loss = -tf.reduce_sum(
#             self.x * tf.log(epsilon+self.x) + 
#             (1-self.x) * tf.log(epsilon+1-self.x), 
#             axis=1
#         )
        recon_loss = tf.reduce_mean(self.recon_loss) 

        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(recon_loss+zloss)
        
        self.losses = {
            'recon_loss': recon_loss,
            'zloss': zloss
        }
        return

    # Execute the forward and the backward pass
    def run_single_step(self,x, label):
        _, losses = self.sess.run(
            [self.train_op, self.losses],
            feed_dict={self.input: x, self.label: label}
        )
        return losses

    # x -> x_hat
    def reconstructor(self, x):
        x_hat, loss = self.sess.run([self.x_hat,self.recon_loss], feed_dict={self.input: x})
        return x_hat, loss
    
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.input: x})
        return z    

    def save_weights(self, path=".models/model.ckpt"):
        self.saver.save(self.sess, path)
        
    def load_weights(self, path=".models/model.ckpt"):
        self.saver.restore(self.sess, path)
        
    def show_vars(self):
        for var in tf.global_variables():
            print(var.name, var.get_shape())
