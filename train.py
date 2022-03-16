# Import librery
import argparse
import sys
import os
import numpy as np
import yaml

from pathlib import Path
from utilitis.datapreprocces import PrepareDataForYolo, getPath, DataGeneratorForYolo

# Define fields
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def parsedata(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--BOX', type=int, default=3, help='how many object could be detected in one grid sell')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--useGenerators', type=bool, nargs='?', const='ram',
                        help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main():
    opt = parsedata()
    model_config_file = ROOT / opt.cfg
    data_config_file = ROOT / opt.data

    config_file = open(data_config_file)
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    train_dir = config["train"]
    val_dir = config["val"]
    list_classes = config["class"]
    classes = len(list_classes)

    train_paths = getPath(train_dir)
    test_paths = getPath(val_dir)

    if not opt.useGenerators:
        X_train, Y_train, train_img_sizes = PrepareDataForYolo(opt.imgsz, data_config_file, model_config_file,
                                                               train_paths, opt.BOX, classes)
        X_test, Y_test, test_img_sizes = PrepareDataForYolo(opt.imgsz, data_config_file, model_config_file, test_paths,
                                                            opt.BOX, classes)
    else:
        train_list_ids = np.arrange(len(train_paths))
        Train_generator = DataGeneratorForYolo(train_list_ids, train_paths, opt.imgsz, model_config_file, opt.BOX,
                                               classes)
        test_list_ids = np.arrange(len(test_paths))
        Test_generator = DataGeneratorForYolo(test_list_ids, test_paths, opt.imgsz, model_config_file, opt.BOX, classes)

if __name__ == "__main__":
    main()
