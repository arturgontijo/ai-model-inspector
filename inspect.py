import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import keras
import cntk
import caffe
import torch


def inspect_tensorflow(model_folder):
    reader = pywrap_tensorflow.NewCheckpointReader(model_folder)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tf.shape(reader.get_tensor(key))


def inspect_keras(model_file, weights_file=""):
    if weights_file:
        with open(model_file, 'r') as f:
            m = keras.models.model_from_json(f.read())
        m.load_weights(weights_file)
    else:
        m = keras.models.load_model(model_file)
        m.summary()
    print("Input: ", m.input)
    print("Shape: ", m.input.shape)


def inspect_cntk(model_file):
    m = cntk.load_model(model_file)
    print(m)


def inspect_caffe(model_file="", prototxt_file=""):
    if model_file:
        net = caffe.Net(prototxt_file, model_file, caffe.TEST)
    else:
        net = caffe.Net(prototxt_file, caffe.TEST)
    print("net.inputs:", net.inputs)
    print("net.inputs(shape:", net.blobs[net.inputs[0]].data.shape)
    print("net.params:", net.params)
    print([(k, v.data.shape) for k, v in net.blobs.items()])


def inspect_torch(model_file):
    m = torch.nn.Module()
    checkpoint = torch.load(model_file, map_location='cpu')
    try:
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            m.load_state_dict(checkpoint['state_dict'])
        else:
            m.load_state_dict(checkpoint)
    except Exception as e:
        print("ERROR:", e)
        m = checkpoint
    finally:
        for k in m.keys():
            print(m[k].shape)


def main():
    opt = input("Framework [tf | keras | cntk | caffe | t7]: ")
    if opt == "" or opt == "tf":
        model_file = input("[{}] Model (Checkpoint Dir): ".format(opt))
        inspect_tensorflow(model_file)
    elif opt == "keras":
        model_file = input("[{}] Model (Architecture JSON): ".format(opt))
        weights_file = input("[{}] Weights (.h5): ".format(opt))
        inspect_keras(model_file, weights_file)
    elif opt == "cntk":
        model_file = input("[{}] Model File: ".format(opt))
        inspect_cntk(model_file)
    elif opt == "caffe":
        model_file = input("[{}] Model File (.caffemodel): ".format(opt))
        prototxt_file = input("[{}] File (.prototxt): ".format(opt))
        inspect_caffe(model_file, prototxt_file)
    elif opt == "t7":
        model_file = input("[{}] Model File (.ckpt|.pt): ".format(opt))
        inspect_torch(model_file)


if __name__ == '__main__':
    main()
