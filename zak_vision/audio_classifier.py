import glob
from datetime import datetime

import librosa
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from tensorflow_core.python.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from tensorflow_core.python.keras.utils import to_categorical

PATH = '/mnt/fad02469-bb9a-4dec-a21e-8b2babc96027/berlin-audio/'


def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(
        data,
        shape=(num_windows, window_size * data.shape[1]),
        strides=((window_size - overlap_size) * data.shape[1] * sz, sz)
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))


def extract_fetures(file_path):
    audio, sr = librosa.load(file_path, 48000)
    mfccs = librosa.feature.mfcc(audio, sr, n_mfcc=64, n_fft=2048, hop_length=1024)
    features = chunk_data(mfccs.T, 64, 48, False)
    return features


def save_data():
    paths = glob.glob(PATH + '*.wav')
    labels = []
    features = []
    for path in paths:
        label = int(path.split('/')[-1].split('.')[0])
        data = extract_fetures(path)
        features.append(data)
        label_data = np.ones((data.shape[0], 1)) * label
        labels.append(label_data)

    labels = np.concatenate(labels, axis=0)
    features = np.concatenate(features, axis=0)

    np.save(PATH + '/labels.npy', labels)
    np.save(PATH + '/features.npy', features)


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(64, 64, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(9, activation='softmax'))

    return model


def load_saved_data():
    x = np.load(PATH + '/features.npy')[:, :, :, np.newaxis]
    y = np.load(PATH + '/labels.npy')[:, 0]
    encoder = LabelEncoder()
    y_cat = to_categorical(encoder.fit_transform(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.2)

    return x_train, x_test, y_train, y_test


def train():
    x_train, x_test, y_train, y_test = load_saved_data()
    model = build_model()
    try:
        model.load_weights(filepath=PATH + '/weights.best.basic_cnn.hdf5')
    except Exception:
        print('No saved checkpoints')
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()

    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100 * score[1]

    print(f'Pre-training accuracy: {accuracy:.4f}%')

    num_epochs = 72
    num_batch_size = 256

    checkpointer = ModelCheckpoint(filepath=PATH + '/weights.best.basic_cnn.hdf5',
                                   verbose=1, save_best_only=True)
    start = datetime.now()

    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
              callbacks=[checkpointer], verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)


def load_model():
    model = build_model()
    model.load_weights(filepath=PATH + '/weights.best.basic_cnn.hdf5')
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model


def show_accuracy():
    x_train, x_test, y_train, y_test = load_saved_data()

    model = load_model()

    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])


def inference():
    le = LabelEncoder()
    le.fit(range(9))

    audio, sr = librosa.load(PATH + '/8.wav', 48000)
    features = librosa.feature.mfcc(audio, sr, n_mfcc=64).T
    start = np.random.randint(0, features.shape[0] - 64)
    x = features[start:start + 64]

    model = load_model()

    predicted_proba_vector = model.predict_proba(x[np.newaxis, :, :, np.newaxis])
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))


def main():
    show_accuracy()


if __name__ == '__main__':
    main()
