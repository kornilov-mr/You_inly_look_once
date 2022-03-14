import numpy as np
import yaml
import os
from PIL import Image

def getPath(dir):
    paths=[]
    for root, dirs, files in os.walk(dir):
        for dir in dirs:
            paths.append(dir)
    return paths

def createLabels(image_dirs,grids,box,classes,imgsz,channels):
    y=np.zeros((len(image_dirs),grids[0],grids[1],box,classes+5),dtype=np.float32)
    x=np.empty((len(image_dirs),imgsz[0],imgsz[1],channels),dtype=np.float32)
    # images_dirs=np.empty((len(iamge_dirs)),dtype=list)
    for image_id,image_path in enumerate(image_dirs):
        label_path=image_path.replace('images','labels')
        with open(label_path) as label:
            contains=np.zeros((grids[0],grids[1]))
            for line in label:
                params=line.split(" ")
                params=[int(_) for _ in params]
                axis1=params[1]//(1/grids[0])
                axis2=params[2]//(1/grids[1])
                y[image_id,axis1,axis2,contains[axis1,axis2],0]=1
                y[image_id, axis1, axis2, contains[axis1, axis2], 1:5] =params[1:5]
                y[image_id, axis1, axis2, contains[axis1, axis2], params[0]+5]=1
                x[image_id]=Image.open(image_path)
                contains[axis1, axis2]+=1
                # images_dirs[image_id]=image_path
    return x,y

def PrepareDataForYolo(imgsz,data_config_path,config_file_path,images_path,box):
    config_file = open(config_file_path)
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    strids = config["all_strids"].split(";")
    grids=imgsz[:]
    for strid in strids:
        grids[0]=int(grids/strid[0])
        grids[1] = int(grids / strid[1])

    data_config = open(data_config_path)
    data_config = yaml.load(data_config, Loader=yaml.FullLoader)
    classes=data_config["classes"]
    x,y=createLabels(images_path,grids,box,len(classes),imgsz,3)
    return x,y





