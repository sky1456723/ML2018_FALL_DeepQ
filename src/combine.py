import sklearn
import sklearn.metrics
import numpy as np
import pandas as pd
import math
import os
from PIL import Image
import torch
import torch.utils
import torch._utils
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import gc
import argparse
import random

class AutoEncoder(torch.nn.Module):
    def __init__(self,c_dim):
        super(AutoEncoder, self).__init__()
        self.G_img_to_label = torch.nn.Sequential(torch.nn.Conv2d(3,128,kernel_size=(7,7),stride=(1,1)), #218x218
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

class Combined_Model(torch.nn.Module):
    def __init__(self, supervised, unsupervised):
        super(Combined_Model, self).__init__()
        self.supervised_model = supervised
        self.unsupervised_model = unsupervised
        self.classifier_input_dim = self.supervised_model.classifier.in_features + \
                                     self.unsupervised_model.to_1d.out_features
        self.classifier = torch.nn.Linear(self.classifier_input_dim, 14)
    def forward(self, x):
        u_out = self.unsupervised_model.G_img_to_label(x)
        u_out = u_out.view(-1, 3*24*24)
        u_out = self.unsupervised_model.to_1d(u_out)
        s_out = self.supervised_model.features(x)
        s_out = F.relu(s_out, inplace=True)
        s_out = F.adaptive_avg_pool2d(s_out, (1, 1)).view(s_out.size(0), -1)
        #print(s_out.shape, u_out.shape)
        s_out = torch.cat((s_out, u_out), dim = 1)
        s_out = self.classifier(s_out)
        s_out = F.sigmoid(s_out)
        return s_out
### DEVICE ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
root_dir = os.path.join(args.root_dir)

train_file = pd.read_csv(os.path.join(args.root_dir,"train.csv"))
label_data = []
unlabel_data = []
for i in train_file.index:
    if type(train_file.loc[i]["Labels"]) != str:
        if math.isnan(train_file.loc[i]["Labels"]):
            pass
            '''
            unlabel_data.append( [ train_file.loc[i]["Image Index"] ])
            '''
    else:
        p = [train_file.loc[i]["Image Index"], train_file.loc[i]["Labels"]]
        label_data.append(p)
        

img_dirs = os.path.join(args.img_dirs)
"""
#Use /255 as normalize
#normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.RandomResizedCrop(224))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
#transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)
def get_dataloader(data_list, transform=None, normalize=None, batch_size = 4, img_dirs = None):
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
            #normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            T.append(transforms.Resize(224))
            T.append(transforms.ToTensor())
            #T.append(normalize)      
            T=transforms.Compose(T)
            
            img = img.convert(mode="RGB")
            img = T(img) / 255
        label = pair[1].split()
        label = np.array([int(c) for c in label])
        part_label.append(label)
        part_data.append(img)

    batch = batch_size
    label_data_x = torch.stack(part_data)
    label_data_y = torch.Tensor(part_label)
    label_dataset = torch.utils.data.TensorDataset(label_data_x, label_data_y)
    label_dataloader = torch.utils.data.DataLoader(dataset = label_dataset,
                                                   batch_size =batch,
                                                   shuffle = True,
                                                   num_workers = 1 )
    del part_data, part_label
    return label_dataloader

def main(args):
    ### Define Model ###
    supervised_model = torchvision.models.densenet201(pretrained = True)
    unsupervised_model = torch.load(args.unsupervised_model_name)
    model = Combined_Model(supervised_model, unsupervised_model).to(device)
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
    
    criterion = torch.nn.BCELoss()

    epoch = args.epoch_number
    model_name = args.model_name
    root_dir = os.path.join(args.root_dir)
    train_file = pd.read_csv(os.path.join(args.root_dir,"train.csv"))
    label_data = []
    unlabel_data = []
    for i in train_file.index:
        if type(train_file.loc[i]["Labels"]) != str:
            if math.isnan(train_file.loc[i]["Labels"]):
                pass
                '''
                unlabel_data.append( [ train_file.loc[i]["Image Index"] ])
                '''
        else:
            p = [train_file.loc[i]["Image Index"], train_file.loc[i]["Labels"]]
            label_data.append(p)


    img_dirs = os.path.join(args.imgs_dir)
    """
    train_data = label_data[len(label_data)//10:]
    val_data = label_data[:len(label_data)//10]
    """
    c = [x for x in range(len(label_data))]
    numb=len(label_data)//10
    selected=random.sample(c,numb)
    val_data =[label_data[i] for i in selected]
    train_data=[label_data[m] for m in c if m not in selected]
    best_auroc = 0 
    for e in range(epoch):
        print("Epoch ",e)
        epoch_loss = 0
        epoch_acc = 0
        model = model.train()
        for part in range(4):
            gc.collect()
            print("Part ",part)
            label_dataloader = get_dataloader(train_data[part*len(train_data)//4:(part+1)*len(train_data)//4],
                                             transform = transformSequence,
                                             normalize = None,
                                             batch_size = args.batch_size,
                                             img_dirs = args.imgs_dir)
            for b_num, (data, label) in enumerate(label_dataloader):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                #preconv_optimizer.zero_grad()
                pred = model(data)
                loss = criterion(pred,label)
                loss.backward()
                optimizer.step()
                print("Batch: ", b_num," loss: ", loss.item(), end='\r')
                #preconv_optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += torch.sum(torch.eq((pred>0.5), label.byte())).item()/14
            del label_dataloader
        
        print("")
        print("Start Validation")
        model = model.eval()
        val_loss = 0
        val_acc = 0
        val_dataloader = get_dataloader(val_data, batch_size = args.batch_size, img_dirs = args.img_dirs)
        ans_list = []
        label_list = []
        for b_num, (data, label) in enumerate(val_dataloader):
            print("Batch: ", b_num, end='\r')
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            val_loss += criterion(pred,label).item()
            for one_row in pred.cpu().data.numpy():
                ans_list.append(one_row)
            for one_row in label.cpu().data.numpy():
                label_list.append(one_row)
        scheduler.step(val_loss)
        #val_loss = val_loss.item()
        del val_dataloader
        auroc = sklearn.metrics.roc_auc_score(np.array(label_list),np.array(ans_list))
        if auroc > best_auroc:
            torch.save(model, model_name)
            torch.save(optimizer.state_dict(), model_name+".optim")
            best_auroc = auroc
        print("")
        print("Epoch loss: ",args.batch_size*epoch_loss/(9*len(label_data)//10) )
        print("Epoch acc: ",epoch_acc/(9*len(label_data)//10) )
        print("Val loss: ",args.batch_size*val_loss/len(val_data))
        print("AUROC: ",auroc )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW3-1 training')
    parser.add_argument('--model_name', type=str, default='default')
    parser.add_argument('--epoch_number', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--supervised_model_name', '-s', type=str)
    parser.add_argument('--unsupervised_model_name', '-u', type=str)
    parser.add_argument('--root_dir', '-r', type=str)
    parser.add_argument('--imgs_dir', '-i', type=str)
    args = parser.parse_args()
    main(args)
