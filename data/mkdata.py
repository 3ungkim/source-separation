import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import librosa


if __name__=="__main__":
    np.random.seed(27)
    path = os.getcwd()

    #index_spker : all speaker index
    #index_spker_test : test speaker index
    index_spker = os.listdir(f"{path}/VCTK-corpus/wav48")
    if '.DS_Store' in index_spker:
        index_spker.remove('.DS_Store')

    index_spker_test = np.random.choice(index_spker, 10)
    index_spker = list(set(index_spker)-set(index_spker_test))

    #df_path_test is dataframe of all data sources path of test dataset
    df_path_test = pd.DataFrame(columns=['path'])

    for i in index_spker_test:
        path_list_test = os.listdir(f"{path}/VCTK-corpus/wav48/{i}")
            
        #for handling unexpected error
        if '.DS_Store' in path_list_test:
            path_list_test.remove('.DS_Store')

        df_temp_path = pd.DataFrame(path_list_test, columns=['path'])
        df_temp_path['path'] = df_temp_path['path'].apply(
            lambda x: os.path.join(f"{path}/VCTK-corpus/wav48/{i}",x)
        )
        df_path_test = pd.concat([df_path_test, df_temp_path], ignore_index=True)

    df_path_test.to_csv("source_path_test.csv")
    n_source_test = len(df_path_test.index)
    #mix_index_test is mixed source index of sources
    mix_index_test = np.random.randint(0, n_source_test, size=(2000, 2))
    np.save(f"{path}/mix_index_test", mix_index_test)

    #df_path is dataframe of all data sources path of train dataset(without test)
    df_path = pd.DataFrame(columns=['path'])

    for i in index_spker[:-1]:
        path_list = os.listdir(f"{path}/VCTK-corpus/wav48/{i}")
        
        #for handling unexpected error
        if '.DS_Store' in path_list:
            path_list.remove('.DS_Store')

        df_temp_path = pd.DataFrame(path_list, columns=['path'])
        df_temp_path['path'] = df_temp_path['path'].apply(
            lambda x: os.path.join(f"{path}/VCTK-corpus/wav48/{i}",x)
        )
        df_path = pd.concat([df_path, df_temp_path], ignore_index=True)
    
    df_path.to_csv("source_path.csv")
    n_source = len(df_path.index)
    #mix_index is mixed source index of sources
    mix_index = np.random.randint(0, n_source, size=(20000, 2))
    np.save(f"{path}/mix_index", mix_index)
    #mix_index = np.load("path/mix_index")


class VCTKDataset(Dataset):
    def __init__(self, path_csv, time=5, sr=8000):
        self.df_path = pd.read_csv(path_csv)
        self.sr = sr 
        self.time = time
        self.eps = 1e-6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __getitem__(self, index):
        path = self.df_path['path'][index]
        len_source = self.sr * self.time
        source, _ = librosa.load(path, sr=self.sr)
        source = torch.from_numpy(source).to(self.device)

        #fix the time of sources
        if len(source)>=len_source:
            source = source[0:len_source]
            return source

        else:
            updated_source = torch.zeros([len_source], device=self.device)
            updated_source = updated_source + self.eps
            crop_first = int(len_source/2 - len(source)/2)
            crop_last = crop_first + len(source)
            updated_source[crop_first:crop_last] = source
            return updated_source

    def __len__(self):
        return len(self.df_path.index)
