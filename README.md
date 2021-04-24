# Covid and Pneumonia Detection from X-Ray Images

This is a TensorFlow based implementation apporach, to identify COVID-19 and Pneumonia(Viral and Bacterial) casses from X-Ray images. The model takes as input X-Ray image of size (224 X 224 X 3) and outputs the probability scores for 3 classes such as (`COVID-19`, `NORMAL`, `PNEUMONIA`).

I used [Transfer learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a) apporach and model used for this is [InceptionV3](https://keras.io/api/applications/inceptionv3/). It has around 311 excluding the top layers.

## Dataset
The datset i used is combined version of 2 different dataset. First one is [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset) for COVID-19 X-Ray images. Another dataset is taken from kaggle [chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) for Pneumonia and Normal images. This second dataset contains both Viral and Baterial Pneumonia images. I combined both category images as one Pneumonia file.

## Dataset Download link:
covid-chestxray: https://github.com/ieee8023/covid-chestxray-dataset<br/>
chest-xray-pneumonia: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia<br/>

### Data Distribution
|  Type | COVID-19 | Normal | Pneumonia | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| Train |  171  |  171  |   171  | 513 |
| Val   | 7 | 7 | 7 | 21
|  Test |   17 | 17  |  17   |   51 |


## Data Augmentation
I have used Tensorflow's [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?version=nightly) for data augmentation wiht `shear_range=0.2`, `zoom_range=0.2` and applied `horizontal flip`.

## Importing the model
I have included the trained model in porject so that you can directly download the model using Tensorflow's [load_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model). And you can start making the prediction.
