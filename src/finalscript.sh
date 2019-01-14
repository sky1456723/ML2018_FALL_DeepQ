python3 ./AE.py --root_dir $1 --imgs_dir $2 -n --model_name AutoEncoder 
for i in {1..13}
do
	python3 ./combine.py -u AutoEncoder --model_name model_$i -r $1 -i $2
done
for j in {1..13}
do
	python3 ./test_combine_model.py ./model_$j ./model_$j.csv $1 $3
done
python3 ./ensemble.py