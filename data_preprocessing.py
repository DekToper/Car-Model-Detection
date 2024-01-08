import os
import shutil
import zipfile
from pathlib import Path
import pandas as pd

from scipy.io import loadmat

train_data_dir = 'stanford_cars/train'
validation_data_dir = 'stanford_cars'
car_devkit = Path('stanford_cars/car_devkit')
car_test = Path('stanford_cars/cars_test')
car_train = Path('stanford_cars/cars_train')

dataset_dir = "stanford_cars/car_train"
train_dir = "stanford_cars/train_set"
validation_dir = "stanford_cars/validation_set"

cars_train_annos = loadmat('stanford_cars/devkit/cars_train_annos.mat')
cars_test_annos = loadmat('stanford_cars/devkit/cars_test_annos.mat')


def get_classes():
    cars_meta = []
    cars_metadata = loadmat('stanford_cars/devkit/cars_meta.mat')
    for x in cars_metadata['class_names'][0]:
        cars_meta.append(x[0])
    cars_classes = pd.DataFrame(cars_meta, columns=['cars_classes_exist_in_data'])
    return cars_classes


def create_validation_data():
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


def create_data_zip():
    cars_classes = get_classes()

    fname = [[x.flatten()[0] for x in i] for i in cars_train_annos['annotations'][0]]
    column_list = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    train_df = pd.DataFrame(fname, columns=column_list)
    train_df['class'] = train_df['class'] - 1
    train_df['fname'] = [car_train / i for i in train_df['fname']]
    train_df.head()

    train_df = train_df.merge(cars_classes, left_on='class', right_index=True)
    train_df = train_df.sort_index()

    zf = zipfile.ZipFile('Stanford Cars Dataset simplified.zip', mode='w')

    try:
        for i in train_df.index:
            print(str(i))
            try:
                name = train_df['cars_classes_exist_in_data'][i]
                file_path = train_df['fname'][i]
                file_name = os.path.basename(train_df['fname'][i])
                short_name = name.split(" ")[0]
                zf.write(file_path, os.path.join(short_name, file_name), zipfile.ZIP_DEFLATED)
            except Exception as exc:
                print(str(exc))
                pass
    finally:
        print('closing')
        zf.close()
