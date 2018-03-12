# Training a Convolutional Neural Network for Multi-label Image Classification

Infer multiple labels for an image using a CNN


## Dependencies

1. [python 3.5.2](https://www.tensorflow.org/install/)
1. [tensorflow 1.4.1](https://www.tensorflow.org/install/)
1. [cadl](https://github.com/pkmital/pycadl)

## Dataset

* RGB JPG images, 256x256 pixels, split into training and test folders

* CSV file with rows of image_names and labels for the training set:

    | image_name | tags                            |
    |------------|---------------------------------|
    | train_0    | haze primary                    |
    | train_1    | agriculture clear primary water |
    | train_2    | clear primary                   |

* CSV file with rows of image_names for test set


## Training

Execute training, validation, and generate labels for test set:

```
python multilabel_cnn.py
```



