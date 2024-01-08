import os

import numpy as np
from keras.optimizers import Adam
from keras.saving.saving_api import load_model
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Шлях до набору даних для тренування та валідації
train_data_dir = 'stanford_cars/train_set'
validation_data_dir = 'stanford_cars/validation_set'


def get_last_version_model():
    file_name = 'car_detection_model'
    for i in range(100):
        if os.path.isfile('models/' + file_name + '_' + str(i) + '.h5'):
            pass
        else:
            return 'models/' + file_name + '_' + str(i) + '.h5'


# Параметри навчання
batch_size = 32
epochs = 40
image_size = (360, 360)
model_file = 'models/car_detection_model_.h5'

# Передобробка та аугментація даних
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=10,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

print(train_generator)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

if os.path.isfile(model_file) is not True:
    base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Заморожуємо ваги базової моделі
    base_model.trainable = False
    # Додавання нових верхніх шарів моделі
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])
else:
    # Завантаження попередньо навченої моделі
    model = load_model(model_file)

# Компіляція моделі
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Тренування моделі
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# Збереження навченої моделі
model.save(get_last_version_model())
