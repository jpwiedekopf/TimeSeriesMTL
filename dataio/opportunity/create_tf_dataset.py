#import tensorflow as tf
import os
import numpy as np
#from tfdataset_adapter import np_to_tfrecords
from tfdataset_adapter import tf_data_record_writer
import requests


def download_proto(tfoutpath):
    protourl = r'https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/example/feature.proto'
    r = requests.get(protourl)
    protoname = os.path.join(tfoutpath, 'feature.proto')
    with open(protoname, 'wb') as f:
        f.write(r.content)
    print(f"Downloaded feature.proto file to {protoname}")


def generate_tf_records(datafile, labelfile, tfoutpath):
    print(f"{datafile} -> {labelfile}")
    datanp = np.load(datafile)
    #examples = datanp.shape[0]
    #steps = datanp.shape[1]
    #channels = datanp.shape[2]
    #data_reshape = np.reshape(datanp, (examples, steps * channels))
    labelsnp = np.load(labelfile)

    prefix = os.path.join(tfoutpath,
                          os.path.basename(datafile).replace("_data.npy", ""))

    # np_to_tfrecords(data_reshape, labelsnp, prefix,
    # verbose=True)
    tf_data_record_writer(datanp, labelsnp, prefix)


if __name__ == "__main__":
    path = r"/data/opportunity/window_labels-mtl/all_sensors/64/"
    tfoutpath = path.replace('window_labels-mtl', 'window_labels-mtl-tf')
    if not os.path.exists(tfoutpath):
        os.makedirs(tfoutpath)
    files = sorted(os.listdir(path))

    for fn in files:
        if "data" in fn:
            datafile = os.path.join(path, fn)
            labelfile = os.path.join(path, fn.replace("_data", "_labels"))
            generate_tf_records(datafile, labelfile, tfoutpath)
