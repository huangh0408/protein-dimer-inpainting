#!/bin/sh
home_dir=`pwd`
cd $home_dir
bash extract_chain.sh
cd $home_dir
matlab -nosplash -nodesktop -r delete_unuseful_col_row
cd $home_dir
python concatenate_matrix.py

