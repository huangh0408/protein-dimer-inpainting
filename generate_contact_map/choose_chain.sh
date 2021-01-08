for i in $(ls ./pdb);do
	name=`echo $i|cut -d '.' -f 1`
	echo "A B" >./chain/${name}_chain.txt
done
