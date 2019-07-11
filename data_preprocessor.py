import numpy as np 
import skimage as ski 
import argparse
import os
import h5py

# https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes
def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description='process data')
    parser.add_argument('-s', type = int, default = 64, help = 'size of image')
    parser.add_argument('-f', type = str, help = 'img path')
    parser.add_argument('-o', type = str, help = 'hdf5 output path')

    opt = parser.parse_args()

    img = ski.io.imread(opt.f)

    left_over = np.array(img.shape) % opt.s
    if(left_over[0] != 0):
        img = img[0:-left_over[0],:,:]
    if(left_over[1] != 0):
        img = img[:,0:-left_over[1],:]
    if(left_over[2] != 0):
        img = img[:,:,0:-left_over[2]]
    
    train_set = cubify(img, (opt.s,opt.s,opt.s))

    hf = h5py.File(opt.o, "w")
    data_set = hf.create_dataset("data", data=train_set.astype(np.float64))

    hf.close()
