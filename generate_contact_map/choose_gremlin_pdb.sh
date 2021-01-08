for i in $(ls ./chain);do
	name=`echo $i|cut -d '_' -f 1`
	cp ./pdb/${name}* ./pdb_gremlin
done
