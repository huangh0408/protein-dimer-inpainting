#CUDA_VISIBLE_DEVICES="3" python evaluate_hid.py --ground_truth_dir /extend2/protein-dimer-inpainting-datasets/Groundtruth  --mode contact --netsize 128
CUDA_VISIBLE_DEVICES="3" python evaluate.py --ground_truth_dir /extend2/protein-dimer-inpainting-datasets/Groundtruth  --mode contact --netsize 128 

