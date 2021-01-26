for i in $(cat ./cd-hit-pdb_homo_list.txt);do
	pdb_str="$i"
	nn=`grep -n "$pdb_str" 0121_masif.txt`
	n=`echo $nn|cut -d ':' -f 1`
	row=$(($n+4))
	awk 'NR=="'"$row"'"{print $0}'  0121_masif.txt>>homodimer_masif_result.txt
done
