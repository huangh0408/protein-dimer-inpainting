CUDA_VISIBLE_DEVICES="0" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/masif_heteromers/ --type=heteromer --mode contact --netsize 256 --batch_size 458
python evaluate_masif_heter.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/masif_heteromers --type=heteromer --mode contact --netsize 256
echo "test masif-ppi "
echo "**************"
echo "*****end******"
mv ./results/test* ./test_test
