import cv2

import tensorflow as tf


def _argment_helper(image):
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [128, 128, 3])
    image = tf.image.resize(image, [227, 227])
    image = tf.math.divide(image, tf.constant(255.0))
    return image


def parse_fn(example_proto):
    "Parse TFExample records and perform simple data augmentation."
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_jpeg(parsed['image_raw'], 3)
    image = _argment_helper(image)
    label = tf.cast(parsed['label'], tf.int64)
    y = tf.one_hot(label, 10)
    return image, y


def input_fn():
    dataset = tf.data.TFRecordDataset('./dataset/tfrecords/train.tfrecords')
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=2)
    dataset = dataset.prefetch(buffer_size=62)
    dataset = dataset.batch(batch_size=32)
    return dataset


dataset = input_fn()
for data in dataset:
    img, label = data
    print(img[0].shape)
    cv2.imshow("decode", img[0].numpy())
    cv2.waitKey(0)
