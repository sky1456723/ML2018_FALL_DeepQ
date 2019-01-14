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
import sys

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

root_dir = os.path.join(sys.argv[3])

test_file = pd.read_csv(os.path.join(root_dir,"test.csv"))
test_data = []
for i in test_file.index:
    test_data.append(test_file.loc[i]["Image Index"])
img_dirs = os.path.join(sys.argv[4])
                      


#normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.Resize(224))
transformList.append(transforms.ToTensor())
#transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)
def get_dataloader_RGB(data_list, transform=None, normalize=None, batch_size = 4):
    part_data = []
    print(len(data_list))
    for i, pair in enumerate(data_list):
        print(i,end='\r')
        img = Image.open(os.path.join(img_dirs, pair))
        img = img.convert(mode="RGB")
        img = transform(img) /255
        part_data.append(img)

    batch = batch_size
    label_data_x = torch.stack(part_data)
    label_dataset = torch.utils.data.TensorDataset(label_data_x)
    label_dataloader = torch.utils.data.DataLoader(dataset = label_dataset,
                                                   batch_size =batch,
                                                   shuffle = False,
                                                   num_workers = 1 )
    del part_data
    return label_dataloader

model_path = sys.argv[1]
"""
model_path = os.path.join("./",model_name)
"""
model = torch.load(model_path).to(device)
model = model.eval()

count = 0

ans_list = []
class_name = open(os.path.join(root_dir,"classname.txt")).readlines()
first_row=["id"]
first_row.extend([name.replace("\n","") for name in class_name])

while True:
    if len(test_data[count*3000:(count+1)*3000]) == 0:
        break
    print("Part ",count)
    dataloader = get_dataloader_RGB(test_data[count*3000:(count+1)*3000],
                                    transform = transformSequence,
                                    batch_size = 4)
    for i, data in enumerate(dataloader):
        print("Batch: ", i, end='\r')
        pred = model(data[0].to(device))
        for one_row in pred.cpu().data.numpy():
            ans_list.append(one_row)
    count += 1
    
            
output_id = np.expand_dims(np.array(test_data),axis=1)
output = np.hstack((output_id, ans_list))
output = np.vstack((first_row, output))
output = pd.DataFrame(output)
output.to_csv(sys.argv[2], index=None, header=None)
