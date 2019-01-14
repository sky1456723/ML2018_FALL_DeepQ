model="model"
wget -O model_1 https://www.dropbox.com/s/bhitm7t9f4i4z4u/combine_model.pkl
wget -O model_2 https://www.dropbox.com/s/dgwsfe1iw9yr4ha/combine_model_2.pkl
wget -O model_3 https://www.dropbox.com/s/hq2j8xqm76nh1sx/combine_model_3.pkl
wget -O model_4 https://www.dropbox.com/s/0aq1tlfch3k35w7/combine_model_densenet201.pkl
wget -O model_5 https://www.dropbox.com/s/q8j94oxqfgfbdfs/combine_model_densenet201_1.pkl?dl=0
wget -O model_6 https://www.dropbox.com/s/shx7j0hk38hltyp/combine_model_densenet201_2.pkl?dl=0
wget -O model_7 https://www.dropbox.com/s/ggsyvg1rummjz2s/test1.model?dl=0
wget -O model_8 https://www.dropbox.com/s/1i5ocs7pvf96fbh/test4_ori?dl=0
wget -O model_9 https://www.dropbox.com/s/oc9tq0gwiq6zhth/test5?dl=0
wget -O model_10 https://www.dropbox.com/s/z0dcz58w38g44nx/default_2?dl=0
for i in {1..10}
do
	python3 ./test_combine_model.py ./model_$i ./model_$i.csv $1 $2
done
python3 ./test_combine_model.py ./model/Combine_more_aug.pkl  model_11.csv $1 $2
python3 ./test_combine_model.py ./model/test_combine.pkl  model_12.csv $1 $2
python3 ./ensemble.py 
