import torch
import torch.utils.data 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from sklearn.model_selection import StratifiedKFold
from adamp import AdamP 


from utils import CustomDataset,MetricMonitor,EarlyStopping
from model import Net
import config as cfg
import os
import fire


def main(root_dir=cfg.root_dir,batch_size=cfg.batch_size,lr=cfg.lr,\
    weight_decay=cfg.weight_decay,n_epochs=cfg.n_epochs,log_dir=cfg.log_dir,k_fold=cfg.k_fold,\
    patience=cfg.patience,steplr_step_size=cfg.steplr_step_size,steplr_gamma=cfg.steplr_gamma,\
    save_dir=cfg.save_dir):
    
    #Seed 
    torch.manual_seed(42)
    np.random.seed(42)

    # Available CUDA
    use_cuda= True if torch.cuda.is_available() else False
    device=torch.device('cuda:0' if use_cuda else 'cpu') # CPU or GPU


    #Transforms
    train_transforms=A.Compose([
        A.RandomCrop(224,224),
        A.ElasticTransform(),         
        A.IAAPerspective(),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    valid_transforms=A.Compose([
        A.CenterCrop(224,224),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()

    ])

    #Dataset
    dataset=CustomDataset(root_dir,transforms=train_transforms)

    
    # Stratified K-fold Cross Validation
    kf=StratifiedKFold(n_splits=k_fold,shuffle=True,random_state=42)

    # Tensorboard Writer
    writer=SummaryWriter(log_dir)


    for n_fold, (train_indices,test_indices) in enumerate(kf.split(dataset.x_data,dataset.y_data),start=1):
        print(f'=====Stratified {k_fold}-fold : {n_fold}=====')

        

        #Dataloader
        train_sampler=torch.utils.data.SubsetRandomSampler(train_indices)
        valid_sampler=torch.utils.data.SubsetRandomSampler(test_indices)
        train_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,sampler=train_sampler)
        valid_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,sampler=valid_sampler)

        #Model,Criterion,Optimizer,scheduler,regularization
        model=Net().to(device)
        criterion=nn.CrossEntropyLoss().to(device)
        optimizer=AdamP(model.parameters(),lr=lr)
        scheduler_steplr=optim.lr_scheduler.StepLR(optimizer,step_size=steplr_step_size,gamma=steplr_gamma)
        regularization=EarlyStopping(patience=patience)

        
        
        #Train
        model.train()
        for epoch in range(n_epochs):
            print(f'Learning rate : {optimizer.param_groups[0]["lr"]}')
            train_metric_monitor=MetricMonitor()
            train_stream=tqdm(train_loader)

            for batch_idx,sample in enumerate(train_stream,start=1):
                img,label=sample['img'].to(device),sample['label'].to(device)
                output=model(img)

                optimizer.zero_grad()
                loss=criterion(output,label)
                _,preds=torch.max(output,dim=1)
                correct=torch.sum(preds==label.data)


                train_metric_monitor.update('Loss',loss.item())
                train_metric_monitor.update('Accuracy',100.*correct/len(img))

                loss.backward()
                optimizer.step()
                train_stream.set_description(
                    f'Train epoch : {epoch} | {train_metric_monitor}'
                )


            # Valid
            valid_metric_monitor=MetricMonitor()
            valid_stream=tqdm(valid_loader)
            model.eval()
            
            with torch.no_grad():
                for batch_idx,sample in enumerate(valid_stream):

                    img,label=sample['img'].to(device),sample['label'].to(device)
                    output=model(img) 
                    
                    loss=criterion(output,label)
                    _,preds=torch.max(output,dim=1)
                    correct=torch.sum(preds==label.data)

                    valid_metric_monitor.update('Loss',loss.item())
                    valid_metric_monitor.update('Accuracy',100.*correct/len(img))
            
            # Tensorboard
            train_loss=train_metric_monitor.metrics['Loss']['avg']
            train_accuracy=train_metric_monitor.metrics['Accuracy']['avg']
            valid_loss=valid_metric_monitor.metrics['Loss']['avg']
            valid_accuracy=valid_metric_monitor.metrics['Accuracy']['avg']
            
            writer.add_scalars(f'{n_fold}-fold Loss',{'train':train_loss,'valid':valid_loss},epoch)
            writer.add_scalars(f'{n_fold}-fold Accuracy',{'train':train_accuracy,'valid':valid_accuracy},epoch)
            
            #Save Model
            if regularization.early_stopping:
                break
            
            regularization.path=os.path.join(save_dir,f'{n_fold}_fold_{epoch}_epoch.pt')
            regularization(val_loss=valid_loss,model=model)

    writer.close()
            
if __name__ == '__main__':
    fire.Fire(main)