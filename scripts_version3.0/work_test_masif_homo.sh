#CUDA_VISIBLE_DEVICES="2" python test_val.py  --input_dir /extend2/data_version_3.0/datasets --type=homodimer --mode contact --netsize 256 --batch_size 216
for i in `seq 57 68`;do
	k=$((i*5))
	python evaluate_masif_homo.py --ground_truth_dir /extend2/data_version_3.0/groundtruth --type=homodimer --mode contact --netsize 256 --interation $k

done

