## TensorFlow-Speech-Recognition-Challenge-with-Conv1d-and-Conv2d
#### https://github.com/kalibianw/TensorFlow-Speech-Recognition-Challenge-with-Conv1d-and-Conv2d
### README Index
1. How can I download TensorFlow Speech Recognition Challenge Dataset?
2. How can I do the preprocessing?
3. How can I train a model?
- - -
#### 1. How can I get TensorFlow Speech Recognition Challenge Dataset?
1. Create a Kaggle account at [Create Kaggle Account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F)
2. Go to [TensorFlow Speech Recognition Challenge data in Kaggle](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)
3. Download All
4. Unzip and remember where you unzip this file.
- - -
#### 2. How can I do the preprocessing?
1. Edit preprocessing_mel.py or preprocessing_mfcc.py or both all as below.<br>
	If you were download and extract "train" folder to "C:\Users\Admin\Document\TensorFlow_Speech_Recognition\", Edit like 
	<pre>
	<code>
	audio_dir_path = YOURPATH + "/TensorFlow_Speech_Recognition/train/audio/"
	</code>
	</pre>
2. Run preprocessing file
- - -
#### 3. How can I train a model?
1. Maybe you can get both file TFSR_80_n_mfcc.npz, TFSR_mel.npz.
2. Just two things what you need to do are Edit below two lines.
<pre>
<code>
ckpt_path = CHECKPOINT_PATH
model_path = MODEL_SAVE_PATH
</code>
</pre>
3. Run training file and done.
