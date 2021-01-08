#!/bin/sh
cd workspace
mkdir temp_dir
for j in $(ls ../pdb);do
	pdb_name=`echo $j|cut -d '.' -f 1`
	echo $pdb_name >>pdb_list.txt
	/home/huanghe/huangh/bioinfo_hh/Complex_Tool/Complex_Tool -i ../pdb/$j -o ${pdb_name}.contact -m 1	
	mkdir ${pdb_name}_temp_dir
	mv *con ${pdb_name}_temp_dir
:<<!
	for k in $(ls ${pdb_name}_temp_dir);do
		len=`expr length $k`
		if [ $len -eq 11 ];then
			name=`echo $k|cut -d '.' -f 1`
			chain=$(echo ${name:6})
			echo $chain >>${pdb_name}_temp_dir/${pdb_name}_chain.txt
		fi
		if [ $len -eq 12 ];then
                        name=`echo $k|cut -d '.' -f 1`
                        chain=$(echo ${name:7})
                        echo $chain >>${pdb_name}_temp_dir/${pdb_name}_chain.txt
                fi
		if [ $len -eq 10 ];then
                        name=`echo $k|cut -d '.' -f 1`
                        chain=$(echo ${name:6})
                        echo $chain >>${pdb_name}_temp_dir/${pdb_name}_chain.txt
                fi
	done
	cp ${pdb_name}_temp_dir/${pdb_name}_chain.txt ../chain
!
	if [ -s ../chain/${pdb_name}_chain.txt ];then
		#awk '{print $1}' ../chain/${pdb_name}_chain.txt >temp_chain_1.txt
		#awk '{print $2}' ../chain/${pdb_name}_chain.txt >temp_chain_2.txt
		s_chain=$(awk 'END{print NR}' ../chain/${pdb_name}_chain.txt)
		if [ ${s_chain} -eq 1 ] ;then
			chain_r=$(awk 'NR==1{print $1}' ../chain/${pdb_name}_chain.txt)
			chain_l=$(awk 'NR==1{print $2}' ../chain/${pdb_name}_chain.txt)
			cat ${pdb_name}_temp_dir/${pdb_name}${chain_r}.con > ../result_contact_matrix/${pdb_name}_${chain_r}_temp.contact_matrix
			cat ${pdb_name}_temp_dir/${pdb_name}${chain_l}.con > ../result_contact_matrix/${pdb_name}_${chain_l}_temp.contact_matrix
			cat ${pdb_name}_temp_dir/${pdb_name}_${chain_r}_${chain_l}.con > ../result_contact_matrix/${pdb_name}_${chain_r}_${chain_l}_temp.contact_matrix
		fi
	fi
	rm -rvf ${pdb_name}_temp_dir
	rm *contact
done
	
