import tensorflow as tf

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image_raw': tf.FixedLenFeature([], tf.string),
                        "coarse_channel": tf.FixedLenFeature([], tf.int64)}
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    
    # Turn your saved image string into an array
    parsed_features['image_raw'] = tf.decode_raw(
        parsed_features['image_raw'], tf.float32)
    
    return parsed_features['image_raw'], parsed_features["coarse_channel"]

# def _shift_resize(image, n=3):
#     if len(image.shape) > 3:
#         image = image.squeeze()
#     batchsize, h, w = image.shape
#     image_out = np.empty((batchsize*n, h, w//2))
#     for i in range(batchsize):
#         starts = np.random.randint(low=5, high=59, size=n)
#         image[i][np.newaxis, ...]
  
class DataSet:

    def __init__(self, filepath, batchsize, SHUFFLE_BUFFER, is_train=True, return_iterator=False):
    
        # This works with arrays as well
        dataset = tf.data.TFRecordDataset(filepath)
        
        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(_parse_function, num_parallel_calls=8)
        
        if is_train:
            # This dataset will go on forever
            dataset = dataset.repeat()

            # Set the number of datapoints you want to load and shuffle 
            dataset = dataset.shuffle(SHUFFLE_BUFFER)
        
        # Set the batchsize
        self.dataset = dataset.batch(batchsize)
        
        if is_train:
            # Create an iterator
            self.iterator = self.dataset.make_one_shot_iterator()
        else:
            self.iterator = self.dataset.make_initializable_iterator()

    def get_next(self):
    
        # Create your tf representation of the iterator
        image, chan = self.iterator.get_next()

        # Bring your picture back in shape
        image = tf.reshape(image, [-1, 16, 128, 1])
        
        # Create a one hot array for your labels
        #label = tf.one_hot(label, NUM_CLASSES)
        return image, chan
