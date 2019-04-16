import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import DataSet
from trainer import *
from model import Autoencoder
import argparse, os


parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--base', type=str, default='/datax/yzhang/models/',
                    help='an integer for the accumulator')
parser.add_argument('--target', type=str, default='./temp_data/test.tfrecords',
                    help='an integer for the accumulator')
parser.add_argument('--epoch', type=int, default=100,
                    help='an integer for the accumulator')
parser.add_argument('--batch_size', type=int, default=128,
                    help='total number of files to use for training')
parser.add_argument('--rec_loss', type=str, default='l2',
                    help='total number of files to use for training')
parser.add_argument('--pre_mode', type=str, default='max',
                    help='total number of files to use for training')
parser.add_argument('--nz', type=int, default=32,
                    help='total number of files to use for training')
args = parser.parse_args()

n_z = 2
batchsize = args.batch_size
SHUFFLE_BUFFER = 128*batchsize
filepath = "./Sband_part1_unnorm.tfrecords"
testpath = "../explore/gan_anomalyd/test_set.tfrecords"
STEPS_PER_EPOC = 100000 // batchsize

trainset = DataSet(filepath, batchsize, SHUFFLE_BUFFER, is_train=True)
testset = DataSet(testpath, batchsize, SHUFFLE_BUFFER, is_train=False)
image, label = trainset.get_next()
image_t, label_t = testset.get_next()
num_sample = 100000
input_dim = 16*128
w = 128
h = 16
datasess = tf.Session()

test_data, test_labels= [], []
datasess.run(testset.iterator.initializer)
while True:
    try:
        d, lab = datasess.run([image_t, label_t])
        test_data.append(d)
        test_labels.append(lab)
    except tf.errors.OutOfRangeError:
        test_data = np.vstack(test_data)
        test_labels = np.concatenate(test_labels, axis=0)
        break

datasess.run(testset.iterator.initializer)



#model_ = Autoencoder(n_z=32, loss_type='l2', use_conv=True)
#model_.load_weights("./models/conv_noise/model.ckpt")

model = trainer(Autoencoder, image, label, n_z=args.nz, num_epoch=30, train_size=num_sample, datasess=datasess, test_data=test_data, alpha_r=3., 
                 loss_type=args.rec_loss, use_conv=True, with_noise=False, bench_model=None, pre_mode=args.pre_mode)
model_dir = './models/conv_log/'
os.makedirs(model_dir)
model.save_weights(model_dir + "model.ckpt")
# for alpha in [1,3,5,10]:
#     # Train a new model
#     model = trainer(Autoencoder, image, label, n_z=args.nz, num_epoch=args.epoch, train_size=num_sample, datasess=datasess, test_data=test_data, alpha_r=alpha, 
#                 loss_type=args.rec_loss, use_conv=True, with_noise=True, bench_model=model_)

#     model_dir = './models/nz_{}/alpha_{}'.format(int(args.nz), int(alpha))
#     os.makedirs(model_dir)
#     model.save_weights(model_dir + "model.ckpt")
