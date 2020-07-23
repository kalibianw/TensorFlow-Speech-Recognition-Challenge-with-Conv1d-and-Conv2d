from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import winsound as ws
import numpy as np

from utils import TrainModule

np_loader = np.load("TFSR_mel.npz")
x_data, y_data = np_loader['x_norm_data'], to_categorical(np_loader['y_data'])
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)

tm_mfcc = TrainModule(result_file_name="TFSR_Mel-spectrogram_training_result",
                      input_shape=np.shape(x_train)[1:],
                      output_shape=np.shape(y_train)[1]
                      )

model1 = tm_mfcc.create_conv1d_model()

ckpt_path = CHECKPOINT_PATH
model_path = MODEL_SAVE_PATH

tm_mfcc.training(
    model=model1,
    x_train=x_train,
    y_train=y_train,
    ckpt_path=ckpt_path,
    model_path=model_path,
    x_test=x_test,
    y_test=y_test
)

ws.Beep(2000, 1000)
