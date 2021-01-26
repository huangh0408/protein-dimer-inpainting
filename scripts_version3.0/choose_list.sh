for i in $(ls ./test_test/images_with_input);do
	name=`echo $i|cut -d "." -f 1`
	echo $name >>pdb_list_test.txt
done
