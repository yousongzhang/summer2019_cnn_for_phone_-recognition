## Acknowledge
1. [mapped from 61 to 48 phonemes] https://github.com/awni/speech
2. [input format for CNN] https://cs224d.stanford.edu/reports/SongWilliam.pdf
3. [CNN model] https://github.com/skyduy/CNN_keras



## Quickly test (TIMIT corpus not required)

```
python predict.py
```

![result](https://github.com/yousongzhang/summer2019_cnn_for_phone_recognition/blob/master/result.png)

## Data

Used The Timit Speech corpus as training and test data.

The data is mapped from 61 to 48 phonemes for training. For final test set
evaluation the 48 phonemes are again mapped to 39. The phoneme mapping is the
standard recipe, the map used here is taken from the [Kaldi TIMIT recipe].



## Basic idea
Apply Convolutional neural network to predict phones on MFCC features of audio record.

CNN input data shape: 40 * 25

40 is MFCC 40 features  (10ms record for MCFF record)
windows size 40 * 25

CNN output data: vector(48)


## Prepare MFCC Data

Once you have the TIMIT data downloaded, edit timit.py file. set TIMIT_PATH

```
python data_prep.py
```

This script will convert the `.WAV` and `.PHE` files to `.mfcc`. The data is mapped from 61 to 48 phonemes in mfcc files.






## Train

load training data, train CNN, show accurate rate of mode predict test result each epoch


```
python train.py
```

## Show Phones Predict Detail

```
python predict.py
```


## Results





[Kaldi TIMIT recipe]: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
