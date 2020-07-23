# for preprocessing
from sidekit.features_extractor import plp
from pydub.utils import mediainfo
import librosa
import os
import re

# for training
from tensorflow.keras import models, layers, activations, optimizers, losses, callbacks, regularizers
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import time


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_key(text):
    return [atoi(c) for c in re.split(r'(/d+)', text)]


class DataModule:
    def __init__(self, audio_dir_path):
        self.audio_count = 0
        self.audio_dir_path = audio_dir_path
        self.dir_list = sorted(os.listdir(self.audio_dir_path), key=natural_key)
        self.each_audio_dir_path = list()
        print("audio_dir_list")
        for i in range(len(self.dir_list)):
            print(audio_dir_path + self.dir_list[i] + "/")
            self.each_audio_dir_path.append(audio_dir_path + self.dir_list[i] + "/")
            audio_list = os.listdir(self.each_audio_dir_path[i])
            audio_list.sort(key=natural_key)
            self.audio_count += len(audio_list)

        print(f"total audio file: {self.audio_count}")

    def audio_mfcc(self, n_mfcc):
        features = list()
        features_norm = list()
        labels = list()
        count = 0
        per = self.audio_count / 1000

        for i in range(len(self.each_audio_dir_path)):
            audio_file_list = sorted(os.listdir(self.each_audio_dir_path[i]), key=natural_key)
            for audio_file_name in audio_file_list:
                audio_file_path = self.each_audio_dir_path[i] + audio_file_name
                audio_file, sr = librosa.load(path=audio_file_path, sr=int(mediainfo(audio_file_path)["sample_rate"]))

                if np.shape(audio_file)[0] < sr:
                    expand_arr = np.zeros(shape=(sr - np.shape(audio_file)[0]), dtype="float32")
                    audio_file = np.append(audio_file, expand_arr)

                elif np.shape(audio_file)[0] > sr:
                    cutted_arr = np.split(ary=audio_file, indices_or_sections=(sr,))
                    audio_file = cutted_arr[0]

                audio_mfcc = librosa.feature.mfcc(y=audio_file, sr=sr, n_mfcc=n_mfcc)
                audio_mfcc_norm = librosa.util.normalize(audio_mfcc)
                features.append(audio_mfcc)
                features_norm.append(audio_mfcc_norm)
                labels.append(i)

                if (count % int(per)) == 0:
                    print(f"{(count / per) / 10}% complete")
                count += 1

        features = np.array(features)
        features_norm = np.array(features_norm)
        labels = np.array(labels)

        return features, features_norm, labels

    def audio_mel(self):
        features = list()
        features_norm = list()
        labels = list()
        count = 0
        per = self.audio_count / 1000

        for i in range(len(self.each_audio_dir_path)):
            audio_file_list = sorted(os.listdir(self.each_audio_dir_path[i]), key=natural_key)
            for audio_file_name in audio_file_list:
                audio_file_path = self.each_audio_dir_path[i] + audio_file_name
                audio_file, sr = librosa.load(path=audio_file_path, sr=int(mediainfo(audio_file_path)["sample_rate"]))

                if np.shape(audio_file)[0] < sr:
                    expand_arr = np.zeros(shape=(sr - np.shape(audio_file)[0]), dtype="float32")
                    audio_file = np.append(audio_file, expand_arr)

                elif np.shape(audio_file)[0] > sr:
                    cutted_arr = np.split(ary=audio_file, indices_or_sections=(sr,))
                    audio_file = cutted_arr[0]

                audio_mel = librosa.feature.melspectrogram(y=audio_file, sr=sr)
                audio_log_mel = librosa.amplitude_to_db(audio_mel)
                audio_mel_norm = librosa.util.normalize(audio_mel)
                features.append(audio_log_mel)
                features_norm.append(audio_mel_norm)
                labels.append(i)

                if (count % int(per)) == 0:
                    print(f"현재 {count}개 완료, {(count / per) / 10}%")
                count += 1

        features = np.array(features)
        features_norm = np.array(features_norm)
        labels = np.array(labels)

        return features, features_norm, labels

    def audio_rplp(self):
        features = list()
        features_norm = list()
        labels = list()
        count = 0
        per = self.audio_count / 1000

        for i in range(len(self.each_audio_dir_path)):
            audio_file_list = sorted(os.listdir(self.each_audio_dir_path[i]), key=natural_key)
            for audio_file_name in audio_file_list:
                audio_file_path = self.each_audio_dir_path[i] + audio_file_name
                audio_file, sr = librosa.load(path=audio_file_path, sr=int(mediainfo(audio_file_path)["sample_rate"]))

                if np.shape(audio_file)[0] < sr:
                    expand_arr = np.zeros(shape=(sr - np.shape(audio_file)[0]), dtype="float32")
                    audio_file = np.append(audio_file, expand_arr)

                elif np.shape(audio_file)[0] > sr:
                    cutted_arr = np.split(ary=audio_file, indices_or_sections=(sr,))
                    audio_file = cutted_arr[0]

                try:
                    audio_rplp = plp(input_sig=audio_file, fs=sr)
                    audio_rplp_norm = librosa.util.normalize(audio_rplp)
                    features.append(audio_rplp)
                    features_norm.append(audio_rplp_norm)
                    labels.append(i)

                except:
                    print("Error occured")

                if (count % int(per)) == 0:
                    print(f"현재 {count}개 완료, {(count / per) / 10}%")
                count += 1

        features = np.array(features)
        features_norm = np.array(features_norm)
        labels = np.array(labels)

        return features, features_norm, labels


class TrainModule:
    def __init__(self, result_file_name, input_shape, output_shape):
        self.result_file_name = result_file_name
        self.result_path = f"{result_file_name}.txt"
        self.input_shape = input_shape
        self.output_shape = output_shape

    def create_conv1d_model(self):
        print(self.input_shape)
        model = models.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Conv1D(filters=128, kernel_size=5, activation=activations.relu, kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(1e-3)
                          ),
            layers.Dropout(rate=0.5),
            layers.MaxPooling1D(),
            layers.Conv1D(filters=256, kernel_size=5, activation=activations.relu, kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(1e-3)
                          ),
            layers.Dropout(rate=0.5),
            layers.Conv1D(filters=256, kernel_size=5, activation=activations.relu, kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(1e-3)
                          ),
            layers.MaxPooling1D(),
            layers.Conv1D(filters=512, kernel_size=5, activation=activations.relu, kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(1e-3)
                          ),
            layers.Dropout(rate=0.5),

            layers.Flatten(),

            layers.Dense(512, activation=activations.relu, kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(1e-3)
                         ),
            layers.Dropout(rate=0.5),
            layers.Dense(256, activation=activations.relu, kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(1e-3)
                         ),
            layers.Dropout(rate=0.5),
            layers.Dense(128, activation=activations.relu, kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(1e-3)
                         ),

            layers.Dense(self.output_shape, activation=activations.softmax)
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_conv2d_model(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=256, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),

            layers.Flatten(),

            layers.Dense(512, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Dense(256, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Dense(128, activation=activations.relu, kernel_initializer="he_normal"),

            layers.Dense(self.output_shape, activation=activations.softmax)
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def training(self, model, x_train, y_train, ckpt_path, model_path, x_test, y_test):
        start_time = time.time()
        kf = KFold(n_splits=5, shuffle=True)
        fhandler = open(self.result_path, mode='w')

        acc_fold = list()
        loss_fold = list()
        fold_no = 1

        for train, valid in kf.split(X=x_train, y=y_train):
            hist = model.fit(
                x=x_train[train], y=y_train[train],
                batch_size=256,
                epochs=500,
                callbacks=[
                    callbacks.ModelCheckpoint(
                        filepath=ckpt_path,
                        verbose=2,
                        save_best_only=True,
                        save_weights_only=True,
                    ),
                    callbacks.EarlyStopping(
                        min_delta=5e-4,
                        patience=25,
                        verbose=2
                    ),
                    callbacks.ReduceLROnPlateau(
                        factor=0.8,
                        patience=5,
                        verbose=2,
                        min_delta=5e-4,
                        min_lr=5e-4
                    )
                ],
                validation_data=(x_train[valid], y_train[valid])
            )

            scores = model.evaluate(x=x_train[valid], y=y_train[valid], verbose=0)
            print(
                f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}"
            )

            acc_fold.append(scores[1] * 100)
            loss_fold.append(scores[0])

            model.load_weights(filepath=ckpt_path)
            model.save(filepath=model_path)

            self.training_visualization(hist=hist.history, fold_no=fold_no)

            fold_no += 1

        fhandler.write(f"Total time: {time.time() - start_time} sec\n")
        fhandler.write("\nScores per fold\n")
        for i in range(0, len(acc_fold)):
            print("--------------------------------------------------------------------------")
            fhandler.write("--------------------------------------------------------------------------\n")
            print(f"> Fold {i + 1} - Loss: {loss_fold[i]} - Accuracy: {acc_fold[i]}%")
            fhandler.write(f"> Fold {i + 1} - Loss: {loss_fold[i]} - Accuracy: {acc_fold[i]}%\n")
        print("--------------------------------------------------------------------------")
        fhandler.write("--------------------------------------------------------------------------\n")
        print("Average scores for all training folds:")
        fhandler.write("\nAverage scores for all training folds:\n")
        print(f"> Accuracy: {np.mean(acc_fold)}% (+- {np.std(acc_fold)})")
        fhandler.write(f"> Accuracy: {np.mean(acc_fold)}% (+- {np.std(acc_fold)})\n")
        print(f"> Loss: {np.mean(loss_fold)} (+- {np.std(loss_fold)})")
        fhandler.write(f"> Loss: {np.mean(loss_fold)} (+- {np.std(loss_fold)})\n")

        test_score = model.evaluate(x=x_test, y=y_test)
        print(
            f"Score for test set: {model.metrics_names[0]} of {test_score[0]}; {model.metrics_names[1]} of {test_score[1] * 100}%")
        fhandler.write(
            f"\nScore for test set: {model.metrics_names[0]} of {test_score[0]}; {model.metrics_names[1]} of {test_score[1] * 100}%\n")

        fhandler.close()

    def training_visualization(self, hist, fold_no):
        localtime = time.localtime()
        tst = str(localtime[0]) + "_" + str(localtime[1]) + "_" + str(localtime[2]) + "_" + str(
            localtime[3]) + "_" + str(localtime[4]) + "_" + str(localtime[5])

        plt.subplot(2, 1, 1)
        plt.plot(hist["acc"], 'b')
        plt.plot(hist["val_acc"], 'g')
        plt.ylim([0, 1])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracies")
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        plt.plot(hist["loss"], 'b')
        plt.plot(hist["val_loss"], 'g')
        plt.ylim([0, 3])
        plt.xlabel("Epoch")
        plt.ylabel("Losses")
        plt.tight_layout()

        fig_path = "fig/" + tst + "_" + str(
            fold_no) + f"_{self.result_file_name}.png"
        plt.savefig(fname=fig_path, dpi=300)
        plt.clf()
