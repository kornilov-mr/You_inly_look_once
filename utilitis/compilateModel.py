#script create file .pt with model discription and model weights
import sys
import yaml
import tensorflow as tf
def batch_norm(inputs, training, data_format,params):
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=params['_BATCH_NORM_DECAY'], epsilon=params['_BATCH_NORM_EPSILON'],
        scale=True, training=training)


def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format)
# def batch_norm(inputs, training, data_format):
#     """Performs a batch normalization using a standard set of parameters."""
#     return tf.layers.batch_normalization(
#         inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
#         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
#         scale=True, training=training)


# class conv2d(nn.Module):
#     def __init__(self,in_channel,out_chanel,kernelx,kernely,padding):
#         super(conv2d, self).__init__()
#         self.conv=nn.Conv2d(in_channel,out_chanel,kernel_size=(kernelx,kernely),padding=padding,bias=False)
#     def forward(self,x):
#         return self.conv(x)
#
# class maxpool(nn.Module):
#     def __init__(self,kernelx,kernely):
#         super(maxpool, self).__init__()
#         self.maxpool=nn.Maxpool2d(stride=(kernelx,kernely))
#     def forward(self,x):
#         return self.maxpool(x)
#
# class dense(nn.Module):
#     def __init__(self,input_units,output_units):
#         super(dense, self).__init__()
#         self.dense=nn.liniar(input_units,output_units)
#         self.activation=nn.LeakyRelu()
#     def forward(self,x):
#         x=self.dense(x)
#         x=self.activation(x)
#         return x
#
# class Yolobilder(nn.Module):
#     def __init__(self,layers):
#         super(Yolobilder, self).__init__()
#         self.layers=layers
#         defined_layers=[]
#         pred_Chanel=3
#         for layerid,layer in enumerate(self.layers):
#             layer_Param=layer.split(",")
#             if layer_Param[0]=="conv":
#                 defined_layers.append(conv2d(pred_Chanel,int(layer_Param[1]),(int(layer_Param[2]),int(layer_Param[3])),layer_Param[4]))
#                 pred_Chanel = int(layer_Param[1])
#             elif  layer_Param[0]=="maxpool":
#                 defined_layers.append(maxpool(int(layer_Param[1]),int(layer_Param[2])))
#             elif layer_Param[0]=="dense":
#                 defined_layers.append(dense(pred_Chanel,int(layer_Param[1])))
#                 pred_Chanel=int(layer_Param[1])
#             else:
#                 print("not defined layer")
#                 return 1;
#
#             self.defined_layers=defined_layers
#     def forward(self,x):
#         for layer in self.defined_layers:
#             x=layer(x)
#         return x

def main():
    config_file_path=sys.argv[1]
    config_file = open(config_file_path)
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    layers=config["layers"].split(";")
    # model=Yolobilder(layers)
    print(sys.argv[1])
if __name__=="__main__":
    main()
