# ML2018_FALL_DeepQ
Machine Learning (2018, Fall) repo for DeepQ competition
## Training
### 套件版本
Pytorch 0.4.1  
numpy 1.15.4  
pandas 0.23.4  
****
### Train
#### Auto-Encoder
執行AE.py以訓練Auto-Encoder模型用於unsupervised learning的部分  
執行方法如下:  
```python3 --model_name [model名字(預設為default)] -e [epoch數量(預設為20)] -b [batch大小(預設為4)] --root [放train.csv跟classname.csv的資料夾] -n(新增模型，或者用-l來載入過去訓練到一半的模型)```   
Auto-Encoder的訓練程度可以由自己決定，在這次的比賽中，我們使用的是訓練到mse_loss=0.16的Auto-Encoder。  
#### Model-Training

#### Ensemble

### Test
執行test_combine_model.py：  
```python test_combine_model.py [model名字] [輸出檔案（請在final/src底下)] [放train.csv跟classname.csv的資料夾] [圖片的資料夾]```  
預設情況下會是 model_編號 對應到 model_編號.csv 為其結果，也請以此方式命名以免ensemble時找不到檔案，    
當12個csv都有的時候ensemble.py才不會出錯  
