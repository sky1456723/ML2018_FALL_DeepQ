import sklearn
import sklearn.metrics
import numpy as np
import pandas as pd
import math
import os
from PIL import Image
import torch

import torch.utils
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import gc
import argparse

class AutoEncoder(torch.nn.Module):
    def __init__(self,c_dim):
        super(AutoEncoder, self).__init__()
        self.G_img_to_label = torch.nn.Sequential(torch.nn.Conv2d(3,128,kernel_size=(6,6),stride=(1,1)), #218x218
                                                  torch.nn.BatchNorm2d(128),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.AvgPool2d((2,2)), #109x109
                                                  torch.nn.Conv2d(128,128,kernel_size=(4,4),stride=(1,1)), #106x106
                                                  torch.nn.BatchNorm2d(128),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Conv2d(128,128,kernel_size=(4,4),stride=(1,1)), #103x103
                                                  torch.nn.BatchNorm2d(128),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Conv2d(128,256,kernel_size=(4,4),stride=(1,1)), #100x100
                                                  torch.nn.BatchNorm2d(256),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.AvgPool2d((2,2)), #50x50
                                                  torch.nn.Conv2d(256,3,kernel_size=(4,4),stride=(2,2)), #24x24
                                                   )
        self.to_1d = torch.nn.Linear(3*24*24, c_dim)
        self.to_2d = torch.nn.Linear(c_dim,128*12*12) #12x12
        self.G_label_to_img = torch.nn.Sequential(torch.nn.ConvTranspose2d(128,256,kernel_size=(4,4),stride=(2,2)), #26x26
                                                  torch.nn.BatchNorm2d(256),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.ConvTranspose2d(256,256,kernel_size=(4,4),stride=(2,2)), #54x54
                                                  torch.nn.BatchNorm2d(256),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.ConvTranspose2d(256,256,kernel_size=(4,4),stride=(2,2)), #110x110
                                                  torch.nn.BatchNorm2d(256),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.ConvTranspose2d(256,100,kernel_size=(4,4),stride=(2,2)), #222x222
                                                  torch.nn.BatchNorm2d(100),
                                                  torch.nn.ConvTranspose2d(100,3,kernel_size=(3,3),stride=(1,1)), #224x224
                                                   )
    def forward(self,x):
        code = self.G_img_to_label(x)
        code = code.view(-1,3*24*24)
        code = self.to_1d(code)
        code = self.to_2d(code)
        code = code.view(-1,128,12,12)
        code = self.G_label_to_img(code)
        return code

def criterion(true,pred):
    loss = F.mse_loss(true,pred)
    loss = torch.sum(loss)
    return loss

### DEVICE ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data ###
root_dir = os.path.join("../","data","ntu_final_data")

train_file = pd.read_csv(os.path.join(root_dir,"medical_images","train.csv"))
label_data = []
unlabel_data = []
for i in train_file.index:
    if type(train_file.loc[i]["Labels"]) != str:
        if math.isnan(train_file.loc[i]["Labels"]):
            unlabel_data.append( [ train_file.loc[i]["Image Index"] ])
            
        

img_dirs = os.path.join(root_dir,"medical_images","images")


### Transform ###
#normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.RandomResizedCrop(224))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
#transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)


def get_dataloader(data_list, transform=None, normalize=None, batch_size = 4):
    part_data = []
    part_label = []
    print(len(data_list))
    for i, pair in enumerate(data_list):
        print(i,end='\r')
        img = Image.open(os.path.join(img_dirs, pair[0]))
        if transform != None:
            img = img.convert(mode="RGB")
            img = transform(img) /255
            #img = np.array(img)/255
            #img = np.transpose(img, axes=[2,0,1])
        else:
            T = []
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            T.append(transforms.Resize(224))
            T.append(transforms.ToTensor())
            T.append(normalize)      
            T=transforms.Compose(T)
            
            img = img.convert(mode="RGB")
            img = T(img)
        part_data.append(img)

    batch = batch_size
    label_data_x = torch.stack(part_data)
    label_dataset = torch.utils.data.TensorDataset(label_data_x)
    label_dataloader = torch.utils.data.DataLoader(dataset = label_dataset,
                                                   batch_size =batch,
                                                   shuffle = True,
                                                   num_workers = 1 )
    del part_data
    return label_dataloader

def main(args):
    ### Define Model ###
    if args.new_model:
        #### Add model ###
        base_model = AutoEncoder(100).to(device)
        optimizer = torch.optim.Adam(base_model.parameters(),lr=0.001)
    elif args.load_model:
        base_model = torch.load(args.model_name).to(device)
        optimizer = torch.optim.Adam(base_model.parameters(),lr=0.001)
        state_dict = torch.load(args.model_name+".optim")
        optimizer.load_state_dict(state_dict)
    
    epoch = args.epoch_number
    model_name = args.model_name

    train_data = unlabel_data

    best_auroc = 0 
    
    for e in range(epoch):
        print("Epoch ",e)
        epoch_loss = 0
        epoch_acc = 0
        
        for part in range(20):
            gc.collect()
            print("Part ",part)
            label_dataloader = get_dataloader(train_data[part*len(train_data)//20:(part+1)*len(train_data)//20],
                                             transform = transformSequence,
                                             normalize = None,
                                             batch_size = args.batch_size)
            for b_num, data in enumerate(label_dataloader):
                data = data[0].to(device)
                optimizer.zero_grad()
                pred = base_model(data)
                loss = criterion(data,pred)
                loss.backward()
                optimizer.step()
                print("Batch: ", b_num," loss: ", loss.item(), end='\r')
                epoch_loss += loss.item()
            del label_dataloader
            torch.save(base_model, model_name)
            torch.save(optimizer.state_dict(), model_name+".optim")
        print("")
        print("Epoch loss: ",args.batch_size*epoch_loss/(len(unlabel_data) ) )
        print("Epoch acc: ",epoch_acc/(9*len(unlabel_data)//10) )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW3-1 training')
    parser.add_argument('--model_name', type=str, default='default')
    parser.add_argument('--epoch_number', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    mutex = parser.add_mutually_exclusive_group(required = True)
    mutex.add_argument('--load_model', '-l', action='store_true', help='load a pre-existing model')
    mutex.add_argument('--new_model', '-n', action='store_true', help='create a new model')

    args = parser.parse_args()
    main(args)