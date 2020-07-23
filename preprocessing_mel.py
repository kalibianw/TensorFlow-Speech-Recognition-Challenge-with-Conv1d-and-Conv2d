from utils import DataModule
import numpy as np

audio_dir_path = YOURPATH + "/TensorFlow_Speech_Recognition/train/audio/"

dm = DataModule(audio_dir_path)

x_data, x_norm_data, y_data = dm.audio_mel()
np.savez_compressed("TFSR_mel.npz", x_data=x_data, x_norm_data=x_norm_data, y_data=y_data)
