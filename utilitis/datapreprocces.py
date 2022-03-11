import numpy
import yaml
def create_yolo_labels(opt):
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    layers = config["layers"].split(";")
    opt.imgsz()
