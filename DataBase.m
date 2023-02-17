load('C:\Users\Admin\Desktop\Train\ExperimentalData');  % load data
InitialScore = [6.5, 7, 5, 5.5, 6, 6.5];
TrainLabels = TrainLabels - InitialScore;
TestLabels = TestLabels - InitialScore;

for i = 1 : 6  % 6 taste
    TrainLabel = TrainLabels(:, i);
    TestLabel = TestLabels(:, i);
    save(strcat('C:/Users/Admin/Desktop/Train/Train-', num2str(i)), 'TrainData', 'TrainLabel', 'TestData', 'TestLabel');
end

