##test masif

CUDA_VISIBLE_DEVICES="1" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/masif_homodimers/ --type=homodimer --mode contact --netsize 256 --batch_size 582
#python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/masif_homodimers --type=homodimer --mode contact --netsize 256
python evaluate_masif_homo.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/masif_homodimers --type=homodimer --mode contact --netsize 256
echo "test masif"
##test val
mv ./results/test* ./test_test
CUDA_VISIBLE_DEVICES="1" python test_val.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/3dcomplex_homodimers --type=homodimer --mode contact --netsize 256 --batch_size 582
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/3dcomplex_homodimers --type=homodimer --mode contact --netsize 256
echo "test 3dcomplex-val"
##test train
#rm -r ./results/test*
mv ./results/test* ./test_val
CUDA_VISIBLE_DEVICES="1" python test_train.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/3dcomplex_homodimers --type=homodimer --mode contact --netsize 256 --batch_size 582
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/3dcomplex_homodimers --type=homodimer --mode contact --netsize 256
echo "test 3dcomplex-train"
mv ./results/test* ./test_train
#CUDA_VISIBLE_DEVICES="0" python test_train.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/3dcomplex_homodimers --type=homodimer --mode contact --netsize 256 --batch_size 582
#python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/3dcomplex_homodimers --type=homodimer --mode contact --netsize 256
#mv ./results/test* ./test_train
##test capri
rm ./data/protein_testset_homodimer_contact_size256.pickle
CUDA_VISIBLE_DEVICES="1" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/casp-capri-12-13-14/ --type=homodimer --mode contact --netsize 256 --batch_size 40
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/casp-capri-12-13-14 --type=homodimer --mode contact --netsize 256
rm ./data/protein_testset_homodimer_contact_size256.pickle
mv ./results/test* ./test_capri
echo "test capri"
CUDA_VISIBLE_DEVICES="1" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/gremlin/ --type=homodimer --mode contact --netsize 256 --batch_size 32
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/gremlin --type=homodimer --mode contact --netsize 256
rm ./data/protein_testset_homodimer_contact_size256.pickle
mv ./results/test* ./test_gremlin
echo "test gremlin"
CUDA_VISIBLE_DEVICES="1" python test_test.py  --input_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/datasets/benchmark5-bound/ --type=homodimer --mode contact --netsize 256 --batch_size 118
python evaluate.py --ground_truth_dir /extendplus/huanghe/dimer_workspace/data_version_2.0/groundtruth/benchmark5-bound --type=homodimer --mode contact --netsize 256
rm ./data/protein_testset_homodimer_contact_size256.pickle
mv ./results/test* ./test_benchmark5-bound
echo "test benchmark5"
