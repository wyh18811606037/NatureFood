load('C:/Users/Admin/Desktop/AAValue/AAValue-5-50-5-100')

NumAA = 9;
NumAAPerSit = size(AAValue, 1);

NumAAAct = 5;
NumSit = combntns(9, NumAAAct);
Sit = combntns(1:9, NumAAAct);
NumAll = NumSit * NumAAPerSit;
Data = zeros(NumAll, NumAA);

for i = 1 : NumSit
    StartP = (i - 1) * NumAAPerSit + 1;
    EndP = i * NumAAPerSit;
    Data(StartP : EndP, Sit(i, :)) = AAValue;
end


time = datestr(now, 30);
for i = 1 : 9
    A1 = (i-1) * NumAll / 9+1;
    A2 = i * NumAll / 9;
    TestData = Data(A1:A2, :);
    save(strcat('C:/Users/Admin/Desktop/TestData/Test-20-100-5-10-', num2str(i), '-', time), 'TestData');
end

%     Now = Sit(i, :);  % Now choosing AA
%     StartP = (i - 1) * NumAAValuePerSit + 1;
%     EndP = i * NumAAValuePerSit;
%     Data(StartP : EndP, Now) = AAValue;

