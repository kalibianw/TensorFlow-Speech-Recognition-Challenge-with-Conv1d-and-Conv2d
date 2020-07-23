from utils import DataModule
import numpy as np

audio_dir_path = YOURPATH + "/TensorFlow_Speech_Recognition/train/audio/"

dm = DataModule(audio_dir_path)

x_data, x_norm_data, y_data = dm.audio_mfcc(n_mfcc=20)
print(np.shape(x_data), np.shape(x_norm_data), np.shape(y_data))
np.savez_compressed("TFSR_20_n_mfcc.npz", x_data=x_data, x_norm_data=x_norm_data, y_data=y_data)

x_data, x_norm_data, y_data = dm.audio_mfcc(n_mfcc=40)
np.savez_compressed("TFSR_40_n_mfcc.npz", x_data=x_data, x_norm_data=x_norm_data, y_data=y_data)

x_data, x_norm_data, y_data = dm.audio_mfcc(n_mfcc=80)
np.savez_compressed("TFSR_80_n_mfcc.npz", x_data=x_data, x_norm_data=x_norm_data, y_data=y_data)
