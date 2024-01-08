import os

import cv2
import numpy as np
from tensorflow import keras

# Завантаження попередньо навченої моделі для розпізнавання автомобілів
model = keras.models.load_model("models/car_detection_model_5.h5")


def get_folder_names(directory):
    folder_names = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            folder_names.append(entry.name)
    return folder_names


def get_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    return files


directory_path = 'stanford_cars/train_set'
classes = get_folder_names(directory_path)

img_nm = get_files_in_directory('Test')

for i in range(img_nm.__len__()):
    # Зчитування зображень
    frame = cv2.imread(img_nm[i])

    # Зміна розміру кадру до 224x224 (розмір вхідного зображення моделі)
    resized_frame = cv2.resize(frame, (224, 224))

    # Нормалізація зображення
    normalized_frame = resized_frame / 255.0

    # Додавання додаткового розміру пакету (batch dimension)
    input_data = np.expand_dims(normalized_frame, axis=0)

    # Розпізнавання об'єкту на зображенні за допомогою моделі
    predictions = model.predict(input_data)
    predicted_class = classes[np.argmax(predictions)]
    print(predictions)
    print(np.argmax(predictions))
    print(classes[np.argmax(predictions)])
    print(str(predictions[0][np.argmax(predictions)] * 100) + '%')

    # Отримання бінарного зображення об'єкта (за допомогою порогування, якщо необхідно)
    ret, binary_image = cv2.threshold(predictions, 0.5, 1, cv2.THRESH_BINARY)

    # Відображення результату на кадрі
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Відображення зображення з результатом
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
