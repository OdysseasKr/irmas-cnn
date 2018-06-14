# irmas-cnn
## Experiments on the IRMAS dataset using Convolutional Neural Networks.

This repository contains code and Jupyter Notebooks of my attempts on the IRMAS dataset.

The IRMAS dataset [[link](https://www.upf.edu/web/mtg/irmas)] is used for musical instrument recognition in audio tracks. It consists of:

- The trainset: Contains 3-second tracks of solo instruments. There are 11 instruments
- The testset: A collection of multi-instrumental audio tracks. Each track is labeled with at least one of the instruments, which is considered its "dominant" sound.

In this project, I have extracted different features from the audio signal, which were fed to a Convolutional Neural Network. The two networks included in the project:

- VGG-16 [[paper](https://arxiv.org/abs/1409.1556)]
- A variation of the YOLO architecture [[paper](https://arxiv.org/abs/1506.02640)]

All of the models are implemented using the new higher level Tensorflow API [[link](https://www.tensorflow.org/programmers_guide/#high_level_apis)].

There are Jupyter Notebooks for two experiments:

- Using Mel-Frequency Cepstrum as feature with YOLO-like CNN [here](https://github.com/OdysseasKr/irmas-cnn/blob/master/Training%20with%20Mel-frequency.ipynb)
- Using several handpicked features with a VGG-16 architecture [here](https://github.com/OdysseasKr/irmas-cnn/blob/master/Training%20with%20handpicked%20features%20and%20VGG-16.ipynb)

### Using the DatasetPreprocessor

The repository includes the DatasetPreprocessor class that extracts features from the raw files of the IRMAS dataset and stores them in easy to use .h5 files. All features are generated using Librosa [[link](https://librosa.github.io/)].

Initialize the DatasetPreprocessor object like this:

``` Python
dp = DatasetPreprocessor('mel')
```

or

``` Python
dp = DatasetPreprocessor('handpicked')
```

The ```mel``` option extract the Mel-Frequency Cepstrum as feature
The ```handpicked``` option extracts the following features:

- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Zero-crossing rate
- RMSE
- MFCC

To generate the train and test sets call
``` Python
dp.generateTrain('path/to/trainset/folder')
dp.generateTest('path/to/testset/folder')
```

Use
``` Python
dp.normalizeGain('path/to/trainset/folder')
```

to normalize the gain of all tracks in a folder, to a specific dB value.
