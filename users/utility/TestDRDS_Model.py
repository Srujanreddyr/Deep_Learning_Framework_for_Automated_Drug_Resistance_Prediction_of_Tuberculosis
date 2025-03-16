from django.conf import settings

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from django.conf import settings
import os
import numpy as np


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(40, 40))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 40, 40, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def start_test(filepath):
    classes = ['Drug Sensitive', 'Drug Resistive']
    # load the image
    file_is = os.path.join(settings.MEDIA_ROOT, filepath)
    img = load_image(file_is)
    model_path = os.path.join(settings.MEDIA_ROOT, 'MyModel.h5')
    model = load_model(model_path)
    result = model.predict(img)
    print("Predicted class Index:", result[0].argmax())
    rstlt = classes[np.argmax(result[0])]
    print("Result:", rstlt)
    return rstlt
