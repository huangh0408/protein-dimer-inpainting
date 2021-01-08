file=dir('./result_contact_matrix/*.contact_matrix');
for n=1:length(file)
	temp=load(['./result_contact_matrix/',file(n).name]);
	[r,l]=size(temp);
	delete_r_vector=ones(1,l)*(-1);
	delete_row=[];
	for i=1:r
		rr=temp(i,:);
		if all(rr==delete_r_vector)
			delete_row=[delete_row;i];
		end
	end
	temp(delete_row,:)=[];
	[r,l]=size(temp);
	delete_l_vector=ones(r,1)*(-1);
	delete_col=[];
        for i=1:l
                ll=temp(:,i);
                if all(ll==delete_r_vector)
                        delete_col=[i,delete_col];
                end
        end
	temp(:,delete_col)=[];
	fid=fopen(['./true_contact_matrix/',file(n).name],'w');
	[m,k]=size(temp);
	for i=1:m
		for j=1:k
			if j==k
				fprintf(fid,'%f\n',temp(i,j));
			else
				fprintf(fid,'%f\t',temp(i,j));
			end
		end
	end
%	fprintf(fid,'%f',temp);
	fclose(fid);
%	save(['./contact_3/',file(n).name],'temp');
end
exit		
