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

### DEVICE ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = os.path.join("../","data","ntu_final_data")

test_file = pd.read_csv(os.path.join(root_dir,"medical_images","test.csv"))
test_data = []
for i in test_file.index:
    test_data.append(test_file.loc[i]["Image Index"])
img_dirs = os.path.join(root_dir,"medical_images","images")
                      


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

input_size = int(sys.argv[3])
model_name = sys.argv[1]
model_path = os.path.join("./",model_name)
model = torch.load(model_path).to(device)
model = model.eval()

count = 0
last = False
ans_list = []
class_name = open(os.path.join(root_dir,"medical_images","classname.txt")).readlines()
first_row=["id"]
first_row.extend([name.replace("\n","") for name in class_name])

while not last:
    if len(test_data[count*3000:(count+1)*3000]) == 0:
        break
    print("Part ",count)
    if input_size == 224:
        dataloader = get_dataloader_RGB(test_data[count*3000:(count+1)*3000],
                                       transform = transformSequence,
                                       batch_size = 16)
        for i, data in enumerate(dataloader):
            print("Batch: ", i, end='\r')
            pred = model.output_act( model(data[0].to(device)) )
            for one_row in pred.cpu().data.numpy():
                ans_list.append(one_row)
    count += 1
    
            
output_id = np.expand_dims(np.array(test_data),axis=1)
output = np.hstack((output_id, ans_list))
output = np.vstack((first_row, output))
output = pd.DataFrame(output)
output.to_csv(sys.argv[2], index=None, header=None)