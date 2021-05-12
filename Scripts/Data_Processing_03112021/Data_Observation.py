# Script visualizes data of surgical actions
## Import library and data
import os
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# ---- File path to the database files  ---- #

# source folder, change based on RUN options (entire script / parts of scripts)
#source_path = os.getcwd()  + '/Data/THoracentesis_03112021'
source_path = os.getcwd()  + '/../..' +  '/Data/THoracentesis_03112021'

# folder name
# file paths to orginal data folders
original_data_folder = ['5 Actions_10032020/OriginalData/ChloraPrep/',
                        '5 Actions_10032020/OriginalData/NeedleInsertion/',
                        '5 Actions_10032020/OriginalData/Dilator/',
                        '5 Actions_10032020/OriginalData/Cut/',
                        '5 Actions_10032020/OriginalData/Tracing/']
# file paths to normalized and modified data folders
processed_data_folder = ['5 Actions_10032020/ChloraPrep/',
                         '5 Actions_10032020/NeedleInsertion/',
                         '5 Actions_10032020/Dilator/',
                         '5 Actions_10032020/Cut/',
                         '5 Actions_10032020/Tracing/']

prepared_data_folder = '5 Actions_10032020/PreparedData/'

# Select single folder for testing - IGNORE
data_folder_ex = '5 Actions_10032020/OriginalData/Tracing/'

save_to_folder = '4sensors_6actions_dataset/Prepared Data/'
# obtain list of files
#csv_list_1 = [f for f in os.listdir(source_path) if fnmatch.fnmatch(f, '*.csv')]
csv_list_1 = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path,f))]