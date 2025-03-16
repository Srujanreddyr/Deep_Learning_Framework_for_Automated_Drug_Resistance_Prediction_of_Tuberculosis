import os
import cv2
import imageio
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import binary_accuracy
import shutil
import matplotlib.pyplot as plt
from django.conf import settings


def extract_target(x):
    target = int(x[-5])
    if target == 0:
        return 'DrugSensitive'
    if target == 1:
        return 'DrugResistive'


def tuberculosis_training_start():
    train_path = os.path.join(settings.MEDIA_ROOT, 'base_dir', 'train_dir')
    valid_path = os.path.join(settings.MEDIA_ROOT, 'base_dir', 'val_dir')
    csv_meta = os.path.join(settings.MEDIA_ROOT, 'tuber_metadata.csv')
    data_tb = pd.read_csv(csv_meta)
    data_tb['age'] = data_tb['age'].replace('5-14 years', 0)
    data_tb['age'] = data_tb['age'].replace('15-24 years', 1)
    data_tb['age'] = data_tb['age'].replace('25-34 years', 2)
    data_tb['age'] = data_tb['age'].replace('35-54 years', 3)
    data_tb['age'] = data_tb['age'].replace('55-74 years', 4)
    data_tb['age'] = data_tb['age'].replace('75+ years', 5)

    # data['age'].value_counts()

    # suicides in different age groups

    x1 = data_tb[data_tb['age'] == 0].sum()
    x2 = data_tb[data_tb['age'] == 1].sum()
    x3 = data_tb[data_tb['age'] == 2].sum()
    x4 = data_tb[data_tb['age'] == 3].sum()
    x5 = data_tb[data_tb['age'] == 4].sum()
    x6 = data_tb[data_tb['age'] == 5].sum()

    # x = pd.DataFrame([x1, x2, x3, x4, x5, x6])
    # x.index = ['5-14', '15-24', '25-34', '35-54', '55-74', '75+']
    # x.plot(kind='bar', color='grey')

    # plt.xlabel('Age Group')
    # plt.ylabel('count')
    # plt.show()
    # data_tb['findings'].value_counts()
    data_tb['target'] = data_tb['study_id'].apply(extract_target)

    data_tb['labels'] = data_tb['target'].map({'DrugSensitive': 0, 'DrugResistive': 1})
    y = data_tb['labels']
    df_train, df_val = train_test_split(data_tb, test_size=0.15, random_state=101, stratify=y)

    num_train_samples = len(df_train)
    num_val_samples = len(df_val)
    train_batch_size = 10
    val_batch_size = 10
    IMAGE_HEIGHT = 40
    IMAGE_WIDTH = 40
    filepath = "model.h5"
    train_steps = np.ceil(num_train_samples / train_batch_size)
    val_steps = np.ceil(num_val_samples / val_batch_size)
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                            batch_size=train_batch_size,
                                            class_mode='categorical')

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                          batch_size=val_batch_size,
                                          class_mode='categorical')

    test_gen = datagen.flow_from_directory(valid_path,
                                           target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                           batch_size=val_batch_size,
                                           class_mode='categorical',
                                           shuffle=False)
    kernel_size = (3, 3)
    pool_size = (2, 2)
    first_filters = 32
    second_filters = 64
    third_filters = 128

    dropout_conv = 0.3
    dropout_dense = 0.3

    model = Sequential()
    model.add(Conv2D(first_filters, kernel_size, activation='relu',
                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(Conv2D(first_filters, kernel_size, activation='relu'))
    model.add(Conv2D(first_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout_dense))
    model.add(Dense(2, activation="softmax"))

    model.summary()

    model.compile(Adam(lr=0.0001), loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                                  verbose=1, mode='max', min_lr=0.00001)

    callbacks_list = [checkpoint, reduce_lr]

    history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  epochs=100, verbose=1,
                                  callbacks=callbacks_list)

    model.metrics_names
    val_loss, val_acc = model.evaluate_generator(test_gen,
                                                 steps=val_steps)

    print('val_loss:', val_loss)
    print('val_acc:', val_acc)
    history_dict = {
        'loss': history.history.get('loss'),
        'accuracy': history.history.get('accuracy'),
        'val_loss': history.history.get('val_loss'),
        'val_accuracy': history.history.get('val_accuracy'),
        'lr': history.history.get('lr')
    }
    test_labels = test_gen.classes
    predictions = model.predict_generator(test_gen, steps=val_steps, verbose=1)
    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FN)
    precision = TP / (TP + FP)
    err = (FN + FP) / (TP + FP + FN + TN)
    recall = (TP / (TP + FN))
    F1 = (2 * precision * recall) / (precision + recall)
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    metrcis = {
        'accuracy': accuracy,
        'sensitivity':sensitivity,
        'specificity':specificity,
        'precision':precision,
        'err':err,
        'F1Score':F1,
        'MCC': MCC

    }
    return history_dict, val_acc, val_loss, metrcis
