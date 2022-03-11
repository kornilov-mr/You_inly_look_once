#script create file .pt with model discription and model weights
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, Sampler

class conv2d(nn.Module):
    def __init__(self,in_channel,out_chanel,kernelx,kernely,padding):
        super(conv2d, self).__init__()
        self.conv=nn.Conv2d(in_channel,out_chanel,kernel_size=(kernelx,kernely),padding=padding,bias=False)
    def forward(self,x):
        return self.conv(x)

class maxpool(nn.Module):
    def __init__(self,kernelx,kernely):
        super(maxpool, self).__init__()
        self.maxpool=nn.Maxpool2d(stride=(kernelx,kernely))
    def forward(self,x):
        return self.maxpool(x)

class dense(nn.Module):
    def __init__(self,input_units,output_units):
        super(dense, self).__init__()
        self.dense=nn.liniar(input_units,output_units)
        self.activation=nn.LeakyRelu()
    def forward(self,x):
        x=self.dense(x)
        x=self.activation(x)
        return x

class Yolobilder(nn.Module):
    def __init__(self,layers):
        super(Yolobilder, self).__init__()
        self.layers=layers
        defined_layers=[]
        pred_Chanel=3
        for layerid,layer in enumerate(self.layers):
            layer_Param=layer.split(",")
            if layer_Param[0]=="conv":
                defined_layers.append(conv2d(pred_Chanel,int(layer_Param[1]),(int(layer_Param[2]),int(layer_Param[3])),layer_Param[4]))
                pred_Chanel = int(layer_Param[1])
            elif  layer_Param[0]=="maxpool":
                defined_layers.append(maxpool(int(layer_Param[1]),int(layer_Param[2])))
            elif layer_Param[0]=="dense":
                defined_layers.append(dense(pred_Chanel,int(layer_Param[1])))
                pred_Chanel=int(layer_Param[1])
            else:
                print("not defined layer")
                return 1;

            self.defined_layers=defined_layers
    def forward(self,x):
        for layer in self.defined_layers:
            x=layer(x)
        return x

def main():
    config_file_path=sys.argv[1]
    config_file = open(config_file_path)
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    layers=config["layers"].split(";")
    model=Yolobilder(layers)
    print(sys.argv[1])
if __name__=="__main__":
    main()
