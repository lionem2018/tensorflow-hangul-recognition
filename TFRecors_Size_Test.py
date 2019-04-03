import numpy as np
import tensorflow as tf
import os, io, time

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, './labels/2350-common-hangul.txt')
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-output')

epochs = 10
num_classes = 2350
batch_size = 100

def Get_dataset_length(train_data_files):
    c = 0
    for fn in train_data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def get_image(serialized_example, num_classes):
    """This method defines the retrieval image examples from TFRecords files.

    Here we will define how the images will be represented (grayscale,
    flattened, floating point arrays) and how labels will be represented
    (one-hot vectors).
    """

    # tf.data.Dataset.map() opens and reads files automatically
    # Just need decoding code for each TFRecord file
    """
    # Convert filenames to a queue for an input pipeline.
    file_queue = tf.train.string_input_producer(files)

    # Create object to read TFRecords.
    reader = tf.TFRecordReader()

    # Read the full set of features for a single example.
    key, example = reader.read(file_queue)
    """

    # Parse the example to get a dict mapping feature keys to tensors.
    # image/class/label: integer denoting the index in a classification layer.
    # image/encoded: string containing JPEG encoded image
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
        })

    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # Decode the JPEG.
    image = tf.image.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float64)
    image = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT])

    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes, dtype=np.uint8))

    return label, image


labels = io.open(DEFAULT_LABEL_FILE, 'r', encoding='utf-8').read().splitlines()

print('Processing data...')

tf_record_pattern = os.path.join(DEFAULT_TFRECORDS_DIR, '%s-*' % 'train')
train_data_files = tf.gfile.Glob(tf_record_pattern)

print(train_data_files)

# Make tf.data.Dataset
# If you want to use one more parameter for decode, use 'lambda' for data.map
dataset = tf.data.TFRecordDataset(train_data_files)
dataset = dataset.map(lambda x: get_image(x, num_classes))
dataset = dataset.repeat(epochs)  # set epoch
dataset = dataset.shuffle(buffer_size=3 * batch_size)  # for getting data in each buffer size data part
dataset = dataset.batch(batch_size)  # set batch size
dataset = dataset.prefetch(buffer_size=1)  # reduce GPU starvation

# Make iterator for dataset
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


start_time = time.time()
init = [tf.global_variables_initializer(), iterator.initializer]
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(init)

train_labels, train_images = sess.run(next_element)

end_time = time.time()
ptime = end_time - start_time

print(f"ptime: {ptime}.")
