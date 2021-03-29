import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import glob
import copy

from collections import defaultdict
from typing import Dict


class CustomDataset(Dataset):
    def __init__(self,root_dir,transforms=None):
        super().__init__()
        self.root_dir =root_dir
        self.transforms=transforms
        self._load_data()
        
    def _load_data(self):
        self.x_data=np.array(glob.glob(f'{self.root_dir}/*/*.jpg'))
        self.y_data=np.array([ int(os.path.split(os.path.split(path)[0])[-1]) for path in self.x_data])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,idx):
        img=cv2.imread(self.x_data[idx])
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        label=self.y_data[idx]
        
        if self.transforms:
            img=self.transforms(image=img)['image']

        sample={'img':img,'label':label}

        return sample

########################################################################
class MetricMonitor():
    def __init__(self,float_precision=3):
        self.float_precision=float_precision
        self._reset()

    def _reset(self):
        self._metrics : Dict[[str,dict]]=defaultdict(lambda : {'val':0,'count':0,'avg':0})
    
    
    def update(self,metric_name,val):
        metric=self._metrics[metric_name]

        metric['val']+=val
        metric['count']+=1
        metric['avg']=metric['val'] / metric['count']

    def __str__(self):
        return '|'.join(
            [f'{metric_name}: {metric["avg"]:.{self.float_precision}f}' for (metric_name,metric) in self.metrics.items()]
        )

    @property
    def metrics(self):
        return copy.deepcopy(self._metrics)


########################################################################
class EarlyStopping():
    def __init__(self,patience=0,verbose=True,path='check_point.pt'):
        """[summary]
        Args:
            patience (int): 몇 epoch 만큼 계속해서 오차가 증가하면 학습을 중단할지 결정한다.
            verbose (bool): validation loss log 를 보여줄지 결정한다.
            path (str, optional): model.pt 를 어디에 저장할지 결정한다.
        """        

        self.patience=patience
        self.verbose=verbose
        self._path=path
        self._step=0
        self._min_val_loss=np.inf
        self._early_stopping=False
    

    def __call__(self,val_loss,model):
        if self._early_stopping: return 

        if self._min_val_loss < val_loss:  #val_loss 증가 
            if self._step >=self.patience:
                self._early_stopping=True
                if self.verbose:
                    print(f'Validation loss increased for {self.patience} epochs...\t Best_val_loss : {self._min_val_loss}')
            elif self._step<self.patience:
                self._step+=1
        else:
            self._step=0
            if self.verbose:
                print(f'Validation loss decreased ({self._min_val_loss:.6f} ---> {val_loss:.6f})\tSaving model..."{self.path}"')
            self._min_val_loss=val_loss
            self.save_checkpoint(model)

    def save_checkpoint(self,model):
        torch.save(model.state_dict(),self.path)

    @property
    def early_stopping(self):
        return self._early_stopping

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self,path):
        self._path=path





