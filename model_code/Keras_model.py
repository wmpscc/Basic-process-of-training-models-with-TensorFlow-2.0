import tensorflow as tf
from tensorflow.python import keras


def _argment_helper(image):
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [128, 128, 3])
    # image = tf.image.resize(image, [227, 227])
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
    dataset = tf.data.TFRecordDataset('../dataset/tfrecords/train.tfrecords')
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=2)
    dataset = dataset.prefetch(buffer_size=62)
    dataset = dataset.batch(batch_size=32)
    return dataset


def creat_model():
    model = keras.Sequential()
    # Input shape: 128*128
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                                  batch_input_shape=(32, 128, 128, 3), kernel_initializer='uniform'))
    model.add(keras.layers.MaxPool2D(strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    # 64*64
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                                  kernel_initializer='uniform'))
    model.add(keras.layers.MaxPool2D(strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    # 32*32
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                                  kernel_initializer='uniform'))
    model.add(keras.layers.MaxPool2D(strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    # 16*16
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
                                  kernel_initializer='uniform'))
    model.add(keras.layers.MaxPool2D(strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    # 8*8
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def train(model, dataset):
    model.summary()
    # save checkpoint
    checkpoint_path = "../save_model/train-{epoch:04d}.ckpt"
    # 每5个epoch保存一次
    callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=5)
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(dataset, epochs=10, steps_per_epoch=100, callbacks=[callback])
    loss, acc = model.evaluate(dataset, steps=10)
    print("Model accuaracy:", acc)


def restore_model():
    model = creat_model()
    path_latest_model = tf.train.latest_checkpoint("../save_model")
    model.load_weights(path_latest_model)
    return model


if __name__ == "__main__":
    dataset = input_fn()
    # model = creat_model()
    # train(model, dataset)

    # 恢复模型
    new_model = restore_model()
    loss, acc = new_model.evaluate(dataset, steps=10)
    print("loss:", loss, "  accuracy:", acc)
