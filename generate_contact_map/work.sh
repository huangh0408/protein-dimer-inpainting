#!/bin/sh
mkdir workspace
mkdir flag_contact_image
mkdir flag_distance_image
mkdir flag_distance_matrix_multi
mkdir flag_distance_matrix_true
mkdir flag_length
mkdir result_contact_matrix
mkdir true_contact_matrix

home_dir=`pwd`
cd $home_dir
bash extract_chain.sh
cd $home_dir
matlab -nosplash -nodesktop -r delete_unuseful_col_row
cd $home_dir
python concatenate_matrix.py

