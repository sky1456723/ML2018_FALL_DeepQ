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

### DEVICE ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = os.path.join("../","data","ntu_final_data")

train_file = pd.read_csv(os.path.join(root_dir,"medical_images","train.csv"))
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
        
        
img_dirs = os.path.join(root_dir,"medical_images","images")

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.RandomResizedCrop(224))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
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
            img = transform(img)
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
    if args.new_model:
        base_model = torchvision.models.densenet121(pretrained = True)
        base_model.classifier = torch.nn.Linear(in_features = base_model.classifier.in_features,
                                                out_features = 14,
                                                bias = True)
        '''
        base_model.add_module("pre_conv",
                      torch.nn.Sequential(
                          torch.nn.Conv2d(3,64, kernel_size=(8,8), stride = (1,1), padding = 1 ), #695x695
                          torch.nn.BatchNorm2d(64),
                          torch.nn.LeakyReLU(),
                          torch.nn.MaxPool2d((3,3)), #231x231
                          torch.nn.Dropout2d(0.3),
                          torch.nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,1)),   #226x226
                          torch.nn.BatchNorm2d(128),
                          torch.nn.LeakyReLU(),
                          torch.nn.Dropout2d(0.3),
                          torch.nn.Conv2d(128, 256, kernel_size=(4,4), stride=(1,1),padding=1),   #224x224
                          torch.nn.BatchNorm2d(256),
                          torch.nn.LeakyReLU(),
                          torch.nn.Dropout2d(0.3), 
                          torch.nn.Conv2d(256, 3, kernel_size=(1,1), stride=(1,1)),   #224x224
                          torch.nn.BatchNorm2d(3),
                          torch.nn.LeakyReLU(),
                          torch.nn.Dropout2d(0.3)
                      )
                     )
        '''
        base_model.add_module("output_act",torch.nn.Sigmoid())
        base_model = base_model.to(device)
        optimizer = torch.optim.Adam(base_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = torch.optim.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
    elif args.load_model:
        base_model = torch.load(args.model_name)
        optimizer = torch.optim.Adam(base_model.parameters(),lr=0.001)
        state_dict = torch.load(args.model_name+".optim")
        optimizer.load_state_dict(state_dict)
        scheduler = torch.optim.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
    #preconv_optimizer = torch.optim.Adam(base_model.pre_conv.parameters(),lr=0.001)
    criterion = torch.nn.BCELoss()

    epoch = args.epoch_number
    model_name = args.model_name

    train_data = label_data[len(label_data)//10:]
    val_data = label_data[:len(label_data)//10]

    best_auroc = 0 
    for e in range(epoch):
        print("Epoch ",e)
        epoch_loss = 0
        epoch_acc = 0
        
        for part in range(4):
            gc.collect()
            print("Part ",part)
            label_dataloader = get_dataloader(train_data[part*len(train_data)//4:(part+1)*len(train_data)//4],
                                             transform = transformSequence,
                                             normalize = None,
                                             batch_size = args.batch_size)
            for b_num, (data, label) in enumerate(label_dataloader):
                print("Batch: ", b_num, end='\r')
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                #preconv_optimizer.zero_grad()
                pred = base_model.output_act( base_model(data)) 
                loss = criterion(pred,label)
                loss.backward()
                optimizer.step()
                #preconv_optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += torch.sum(torch.eq((pred>0.5), label.byte())).item()/14
            del label_dataloader
        
        print("")
        print("Start Validation")
        val_loss = 0
        val_acc = 0
        val_dataloader = get_dataloader(val_data, batch_size = args.batch_size)
        ans_list = []
        label_list = []
        for b_num, (data, label) in enumerate(val_dataloader):
            print("Batch: ", b_num, end='\r')
            data = data.to(device)
            label = label.to(device)
            pred = base_model.output_act( base_model(data) )
            val_loss += criterion(pred,label)
            for one_row in pred.cpu().data.numpy():
                ans_list.append(one_row)
            for one_row in label.cpu().data.numpy():
                label_list.append(one_row)
        scheduler.step(val_loss)
        val_loss = val_loss.item()
        del val_dataloader
        auroc = sklearn.metrics.roc_auc_score(np.array(label_list),np.array(ans_list))
        if auroc > best_auroc:
            torch.save(base_model, model_name)
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
    mutex = parser.add_mutually_exclusive_group(required = True)
    mutex.add_argument('--load_model', '-l', action='store_true', help='load a pre-existing model')
    mutex.add_argument('--new_model', '-n', action='store_true', help='create a new model')

    args = parser.parse_args()
    main(args)