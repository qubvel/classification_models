# Classification models Zoo
Pretrained classification models for Keras

### Models: 
- [ResNet](https://arxiv.org/abs/1512.03385) models converted from MXNet:
  - [ResNet18](https://github.com/qubvel/classification_models/blob/master/imgs/graphs/resnet18.png)
  - [ResNet34](https://github.com/qubvel/classification_models/blob/master/imgs/graphs/resnet34.png)
  - [ResNet50](https://github.com/qubvel/classification_models/blob/master/imgs/graphs/resnet50.png)
  - ResNet101
  - ResNet152
- [ResNeXt](https://arxiv.org/abs/1611.05431) models converted from MXNet:
  - ResNeXt50
  - ResNeXt101
- [ResNet](https://arxiv.org/abs/1512.03385) models with [squeeze and excitation](https://arxiv.org/abs/1709.01507) block:
  - SE-ResNet50 (converted from [PyTorch](https://github.com/Cadene/pretrained-models.pytorch))
  - SE-ResNet101 (converted from [PyTorch](https://github.com/Cadene/pretrained-models.pytorch))
  - SE-ResNet152 (converted from [PyTorch](https://github.com/Cadene/pretrained-models.pytorch))
- [ResNeXt](https://arxiv.org/abs/1611.05431) models with [squeeze and excitation](https://arxiv.org/abs/1709.01507) block:
  - SE-ResNeXt50 (converted from [PyTorch](https://github.com/Cadene/pretrained-models.pytorch))
  - SE-ResNeXt101 (converted from [PyTorch](https://github.com/Cadene/pretrained-models.pytorch))
- [SENet154](https://arxiv.org/abs/1709.01507) (converted from [PyTorch](https://github.com/Cadene/pretrained-models.pytorch))
  
| Model           |Acc@1|Acc@5|Time*|
|-----------------|:---:|:---:|:---:|
|densenet121      |74.67|92.04|27.66|
|densenet169      |75.85|92.93|33.71|
|densenet201      |77.13|93.43|42.40|
|inceptionresnetv2|80.03|94.89|54.77|
|inceptionv3      |77.55|93.48|38.94|
|mobilenet        |70.36|89.39|15.50|
|mobilenetv2      |71.63|90.35|18.31|
|nasnetlarge      |82.12|95.72|116.53|
|nasnetmobile     |74.04|91.54|27.73|
|resnet101        |76.58|93.10|33.03|
|resnet101v2      |71.93|90.41|28.80|
|resnet152        |76.66|93.08|42.37|
|resnet152v2      |72.29|90.61|41.09|
|resnet18         |68.24|88.49|16.07|
|resnet34         |72.17|90.74|17.37|
|resnet50         |74.81|92.38|22.62|
|resnet50v2       |69.73|89.31|19.56|
|resnext101       |78.48|94.00|60.07|
|resnext50        |77.36|93.48|37.57|
|senet154         |81.06|95.24|137.36|
|seresnet101      |77.92|94.00|32.55|
|seresnet152      |78.34|94.08|47.88|
|seresnet50       |76.44|93.02|23.64|
|seresnext101     |79.88|94.87|62.80|
|seresnext50      |78.74|94.30|38.29|
|vgg16            |70.79|89.74|24.95|
|vgg19            |70.89|89.69|24.95|
|xception         |78.87|94.20|42.18|

### Installation
PyPi package:
```bash
$ pip install image-classifiers
```
Latest version:
```bash
$ pip install git+https://github.com/qubvel/classification_models.git
```

### Example  

Imagenet inference example:  
```python
import numpy as np
from skimage.io import imread
from skimage.transfrom import resize
from keras.applications.imagenet_utils import decode_predictions

from classification_models import ResNet18
from classification_models.resnet import preprocess_input

# read and prepare image
x = imread('./imgs/tests/seagull.jpg')
x = resize(x, (224, 224)) * 255    # cast back to 0-255 range
x = preprocess_input(x)
x = np.expand_dims(x, 0)

# load model
model = ResNet18(input_shape=(224,224,3), weights='imagenet', classes=1000)

# processing image
y = model.predict(x)

# result
print(decode_predictions(y))
```

Model fine-tuning example:
```python
import keras
from classification_models import ResNet18

# prepare your data
X = ...
y = ...

n_classes = 10

# build model
base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = keras.layers.AveragePooling2D((7,7))(base_model.output)
x = keras.layers.Dropout(0.3)(x)
output = keras.layers.Dense(n_classes)(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

# train
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y)
```
