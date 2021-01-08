for i in $(ls chain_temp);do
	pdb_name=`echo $i|cut -d '.' -f 1`
	a=$(awk 'NR==1{print $1}' ./chain_temp/$i)
	b=$(awk 'NR==2{print $1}' ./chain_temp/$i)	
	echo $a $b >>./chain/$i
done
