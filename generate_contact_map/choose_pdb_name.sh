for j in $(ls /extendplus/huanghe/dimer_workspace/model_training/Study/data/protein_contact_3dcomplex_new/protein_eval_gt);do
        name=`echo $j|cut -d '.' -f 1`
        s1=`echo $name|cut -d '_' -f 1`
        s2=`echo $name|cut -d '_' -f 2`
        s3=`echo $name|cut -d '_' -f 3`
        s4=`echo $name|cut -d '_' -f 4`
        pdb_name=${s1}_${s2}
        chain_1=${s3}
        chain_2=${s4}
	cat temp4.txt |grep $pdb_name >temp.txt
	s=$(awk 'END{print NR}' temp.txt)
	if [ $s -gt 0 ];then
		echo $pdb_name >>cd-hit-pdb_list_3dcomplex_test.txt
	fi
done

