Interval = 0.05;%
End = 0.2;%
NumAAAct = 5;
AAValue = zeros(round((End / Interval + 1)^NumAAAct), NumAAAct);
i = 0;
MaxAA = 0.3 ;
for i1 = 0 : Interval : End
for i2 = 0 : Interval : End
for i3 = 0 : Interval : End
for i4 = 0 : Interval : End
for i5 = 0 : Interval : End
i = i + 1;
AAValue(i, :) = [i1, i2, i3, i4, i5];
end
end
end
end
end
p=sum(AAValue, 2);
A=find(p<=MaxAA);%
B=AAValue(A(:),:);
save(strcat('C:/Users/Admin/Desktop/AAValue/AAValue-', num2str(Interval*100), '-', num2str(End*100), '-', num2str(NumAAAct), '-', num2str(MaxAA*100)), 'B');




