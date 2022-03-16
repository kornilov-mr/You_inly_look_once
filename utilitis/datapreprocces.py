import numpy as np
import yaml
import os
from PIL import Image
from tensorflow.keras.utils import Sequence as tfGenerator


def getPath(dir):
    paths = []
    for root, dirs, files in os.walk(dir):
        for dir in dirs:
            paths.append(dir)
    return paths


def createLabels(image_dirs, grids, box, classes, imgsz, channels):
    y = np.zeros((len(image_dirs), grids[0], grids[1], box, classes + 5), dtype=np.float32)
    x = np.empty((len(image_dirs), imgsz[0], imgsz[1], channels), dtype=np.float32)
    img_sizes = np.empty((len(image_dirs), 2))
    for image_id, image_path in enumerate(image_dirs):
        label_path = image_path.replace('images', 'labels')
        with open(label_path) as label:
            contains = np.zeros((grids[0], grids[1]))
            for line in label:
                params = line.split(" ")
                params = [int(_) for _ in params]
                axis1 = params[1] // (1 / grids[0])
                axis2 = params[2] // (1 / grids[1])
                y[image_id, axis1, axis2, contains[axis1, axis2], 0] = 1
                y[image_id, axis1, axis2, contains[axis1, axis2], 1:5] = params[1:5]
                y[image_id, axis1, axis2, contains[axis1, axis2], params[0] + 5] = 1
                contains[axis1, axis2] += 1

                image = Image.open(image_path)
                x[image_id] = image.resize((imgsz[0], imgsz[1]), Image.ANTIALIAS)
                img_sizes[image_id] = image.size
    return x, y, img_sizes


def Getgrids(model_config_path, imgsz):
    config_file = open(model_config_path)
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    strids = config["all_strids"].split(";")
    grids = imgsz[:]
    for strid in strids:
        grids[0] = int(grids / strid[0])
        grids[1] = int(grids / strid[1])
    return grids


def PrepareDataForYolo(imgsz, model_config_path, images_path, box, classes):
    grids = Getgrids(model_config_path, imgsz)
    x, y, img_sizes = createLabels(images_path, grids, box, classes, imgsz, 3)
    return x, y, img_sizes


class DataGeneratorForYolo(tfGenerator):
    def __init__(self, list_ids, image_paths, imgsz, model_config_path, BOX, CLASSES, batch_size=10):
        self.list_ids = list_ids
        self.BOX = BOX
        self.CLASSES = CLASSES
        self.imgsz = imgsz
        self.image_paths = image_paths
        self.label_paths = image_paths.replace('images', 'labels')
        self.batch_size = batch_size
        self.grids = Getgrids(model_config_path, imgsz)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y, img_sizes = self.__data_generation(list_IDs_temp)
        return {"Train": X, "Test": Y, "Size": img_sizes}

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.imgsz[0], self.imgsz[1], 3))
        y = np.empty((self.batch_size, self.grids[1], self.grids[0], self.BOX, self.CLASSES + 5))
        img_sizes = np.empty((self.batch_size, 2))
        for i, ID in enumerate(list_IDs_temp):
            image = Image.open(self.image_path[ID])
            X[i] = image.resize((self.imgsz[0], self.imgsz[1]), Image.ANTIALIAS)
            img_sizes[i] = image.size
            with open(self.label_paths[ID]) as label:
                contains = np.zeros((self.grids[0], self.grids[1]))
                for line in label:
                    params = line.split(" ")
                    params = [int(_) for _ in params]
                    axis1 = params[1] // (1 / self.grids[0])
                    axis2 = params[2] // (1 / self.grids[1])
                    y[i, axis1, axis2, contains[axis1, axis2], 0] = 1
                    y[i, axis1, axis2, contains[axis1, axis2], 1:5] = params[1:5]
                    y[i, axis1, axis2, contains[axis1, axis2], params[0] + 5] = 1
                    contains[axis1, axis2] += 1
        X = X / 255
        return X, y, img_sizes
