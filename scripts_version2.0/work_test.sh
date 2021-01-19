#test masif
'''
CUDA_VISIBLE_DEVICES="1" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/masif_heteromers/ --type=heteromer --mode contact --netsize 256 --batch_size 458
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/masif_heteromers --type=heteromer --mode contact --netsize 256
echo "test masif-ppi "
echo "**************"
echo "*****end******"
mv ./results/test* ./test_test
#test val
CUDA_VISIBLE_DEVICES="1" python test_val.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/3dcomplex_heteromers --type=heteromer --mode contact --netsize 256 --batch_size 550
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/3dcomplex_heteromers --type=heteromer --mode contact --netsize 256
echo "test 3dcomplex-val "
echo "**************"
echo "*****end******"
mv ./results/test* ./test_val
#test train
CUDA_VISIBLE_DEVICES="1" python test_train.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/3dcomplex_heteromers --type=heteromer --mode contact --netsize 256 --batch_size 550
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/3dcomplex_heteromers --type=heteromer --mode contact --netsize 256
echo "test 3dcomplex-train "
echo "**************"
echo "*****end******"
mv ./results/test* ./test_train
rm ./data/protein_testset_heteromer_contact_size256.pickle
'''
CUDA_VISIBLE_DEVICES="1" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/casp-capri-12-13-14/ --type=heteromer --mode contact --netsize 256 --batch_size 40
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/casp-capri-12-13-14 --type=heteromer --mode contact --netsize 256
rm ./data/protein_testset_heteromer_contact_size256.pickle
mv ./results/test* ./test_capri
echo "test capri"
CUDA_VISIBLE_DEVICES="1" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/gremlin/ --type=heteromer --mode contact --netsize 256 --batch_size 32
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/gremlin --type=heteromer --mode contact --netsize 256
rm ./data/protein_testset_heteromer_contact_size256.pickle
mv ./results/test* ./test_gremlin
echo "test gremlin"
CUDA_VISIBLE_DEVICES="1" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/benchmark5-bound/ --type=heteromer --mode contact --netsize 256 --batch_size 118
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/benchmark5-bound --type=heteromer --mode contact --netsize 256
rm ./data/protein_testset_heteromer_contact_size256.pickle
mv ./results/test* ./test_benchmark5-bound
echo "test benchmark5"
