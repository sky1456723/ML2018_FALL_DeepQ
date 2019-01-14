# ML2018_FALL_DeepQ
Machine Learning (2018, Fall) repo for DeepQ competition
## 套件版本
* Pytorch 0.4.1  
* numpy 1.15.4  
* pandas 0.23.4
## Training
執行finalscript.sh以開始訓練。執行方法如下:
```bash finalscript.sh [放train.csv跟classname.csv的資料夾] [圖片的資料夾]```
如果執行失敗或者想要更細部的操作，請參閱下個條目。
## Training(在bash失敗或想要細部調整的情況下)
### Auto-Encoder
執行AE.py以訓練Auto-Encoder模型用於unsupervised learning的部分。執行方法如下:  
```python3 --model_name [model名字(預設為default)] -e [epoch數量(預設為20)] -b [batch大小(預設為4)] --root [放train.csv跟classname.csv的資料夾] -n(新增模型，或者用-l來載入過去訓練到一半的模型)```   
Auto-Encoder的訓練程度可以由自己決定。  
在這次的比賽中，我們使用的是訓練到mse_loss=0.16的Auto-Encoder。  
****
### Model-Training  
執行combine.py以訓練supervised learning與Auto-Encoder的共同訓練。執行方法如下:  
```python3 combine.py --model_name [請以model_(1到12)依序命名] -e [epoch數量(預設為20)] -b [batch大小(預設為8，可以依照硬體強度調大)] -u [訓練好的Auto-Encoder之model名字] -r [放train.csv跟classname.csv的資料夾] -i [圖片的資料夾]```  
例如python3 combine.py --model_name model_1 -e 20 -b 8 -u default -r ntu_final_data -i ntu_final_data/medical_images  
請執行12次以達成ensemble的目標大小，或者手動進入ensemble修改model數量上限。  
****
### Ensemble  
執行ensemble.py以進行bagging方法的實踐。執行方法如下:
```python3 ensemble.py```
本次比賽中，我們使用12個model的ensemble，少量增加Model數量可以增加一點AUROC數值。

## Testing
執行test_combine_model.py：    
```python3 test_combine_model.py [model名字] [輸出檔案（請在final/src底下)] [放train.csv跟classname.csv的資料夾] [圖片的資料夾]```  
預設情況下會是 model_編號 對應到 model_編號.csv 為其結果，也請以此方式命名以免ensemble時找不到檔案，    
當12個csv都有的時候ensemble.py才不會出錯。 
