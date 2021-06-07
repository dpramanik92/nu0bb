import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from ndbd import *
import os
import shutil

def Create_output_folders(_outpath):

    isFile = os.path.isdir(_outpath)


    if(isFile!=True):
        print('Output Folder does not exist..')
        s = [1.0]
        for i in tqdm(s):

            plot_path = _outpath +'/plots'
            data_path = _outpath + '/data_files'
            os.makedirs(plot_path)
            os.makedirs(data_path)

        print('Output folder created with path:',_outpath)
    else:
        print('Output folder already exists at path: ',_outpath)

def Clean_output_folders(_output):

    if(os.path.isdir(_outpath)!=False):
        shutil.rmtree(_outpath)
        print('Output path removed...')
    else:
        print('Path does not exists')



outpath = os.path.expanduser('~/Documents/nu0bb/data/')
Create_output_folders(outpath)
out_data = outpath + '/data_files'
out_plot = outpath + '/plots'
