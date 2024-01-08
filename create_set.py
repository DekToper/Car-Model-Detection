import os
import shutil

dataset_dir = "stanford_cars/car_train"
train_dir = "stanford_cars/train_set"
validation_dir = "stanford_cars/validation_set"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

split_ratio = 0.8

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        train_class_dir = os.path.join(train_dir, class_name)
        validation_class_dir = os.path.join(validation_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(validation_class_dir, exist_ok=True)

        file_names = os.listdir(class_dir)
        split_index = int(len(file_names) * split_ratio)
        train_files = file_names[:split_index]
        validation_files = file_names[split_index:]

        for file_name in train_files:
            src = os.path.join(class_dir, file_name)
            dst = os.path.join(train_class_dir, file_name)
            shutil.copy(src, dst)

        for file_name in validation_files:
            src = os.path.join(class_dir, file_name)
            dst = os.path.join(validation_class_dir, file_name)
            shutil.copy(src, dst)
