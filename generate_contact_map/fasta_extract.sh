for j in $(ls ./flag_contact_image);do
        name=`echo $j|cut -d '.' -f 1`
	s1=`echo $name|cut -d '_' -f 1`
	s2=`echo $name|cut -d '_' -f 2`
	s3=`echo $name|cut -d '_' -f 3`
        s4=`echo $name|cut -d '_' -f 4`
	pdb_name=${s1}_${s2}
	chain_1=${s3}
	chain_2=${s4}
	awk '$1=="'"ATOM"'"&&$5=="'"$chain_1"'"{print $0}' ./pdb/${pdb_name}.pdb >${pdb_name}_${chain_1}.pdb
	awk '$1=="'"ATOM"'"&&$5=="'"$chain_2"'"{print $0}' ./pdb/${pdb_name}.pdb >${pdb_name}_${chain_2}.pdb
	python pdb2fasta_file.py ${pdb_name}_${chain_1}.pdb
	python pdb2fasta_file.py ${pdb_name}_${chain_2}.pdb	
	fasta_name_1=">${pdb_name}_${chain_1}"
	fasta_name_2=">${pdb_name}_${chain_2}"
	content1=$(awk '{print $1}' ${pdb_name}_${chain_1}.fasta)
	content2=$(awk '{print $1}' ${pdb_name}_${chain_2}.fasta)
	echo $fasta_name_1 >>fasta_all_chain.txt
	echo $content1 >>fasta_all_chain.txt
	echo $fasta_name_2 >>fasta_all_chain.txt
        echo $content2 >>fasta_all_chain.txt
	rm ${pdb_name}_${chain_1}.fasta
	rm ${pdb_name}_${chain_2}.fasta
	rm ${pdb_name}_${chain_1}.pdb
	rm ${pdb_name}_${chain_2}.pdb
done
	
