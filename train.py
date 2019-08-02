"""
    File name: train.py
    Function Des:

    ~~~~~~~~~~

    author: Skyduy <cuteuy@gmail.com> <http://skyduy.me>

"""
import os
import numpy as np
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import load_data

# from core.utils import load_data, APPEARED_LETTERS
#
#
# def prepare_data(folder):
#     print('... loading data')
#     letter_num = len(APPEARED_LETTERS)
#     data, label = load_data(folder)
#     data_train, data_test, label_train, label_test = \
#         train_test_split(data, label, test_size=0.1, random_state=0)
#     label_categories_train = to_categorical(label_train, letter_num)
#     label_categories_test = to_categorical(label_test, letter_num)
#     return (data_train, label_categories_train,
#             data_test, label_categories_test)

def prepare_data():
    return (np.array(load_data.train_x), np.array(load_data.train_y),
                np.array(load_data.test_x), np.array(load_data.test_y))


def build_model():
    print('... construct network')


    inputs = layers.Input((25, 40, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(load_data.target), activation='softmax')(x)

    return Model(inputs=inputs, outputs=out)


def train(pic_folder, weight_folder):
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    x_train, y_train, x_test, y_test = prepare_data()
    model = build_model()

    print('... compile models')
    model.compile(
        optimizer='adadelta',
        loss=['categorical_crossentropy'],
        metrics=['accuracy'],
    )

    print('... begin train')

    check_point = ModelCheckpoint(
        os.path.join(weight_folder, '{epoch:02d}.hdf5'))

    class TestAcc(Callback):
        def on_epoch_end(self, epoch, logs=None):
            weight_file = os.path.join(
                weight_folder, '{epoch:02d}.hdf5'.format(epoch=epoch + 1))
            model.load_weights(weight_file)
            out = model.predict(x_test, verbose=1)
            predict = np.array([np.argmax(i) for i in out])
            answer = np.array([np.argmax(i) for i in y_test])
            acc = np.sum(predict == answer) / len(predict)
            print('Single phone test accuracy: {:.2%}'.format(acc))
            print('----------------------------------\n')

    model.fit(
        x_train, y_train, batch_size=128, epochs=100,
        validation_split=0.1, callbacks=[check_point, TestAcc()],
    )


if __name__ == '__main__':
    train(
        pic_folder=r'',
        weight_folder=r'CNN_keras_models'
    )
