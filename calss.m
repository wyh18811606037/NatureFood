clear

%%终末の版
%% Desired Flavor Score
FlavorInitial = [6.5, 7, 5, 5.5, 6, 6.5];%初始风味
FlavorTarget =  [6.5, 7.5, 6.8, 5.5, 5, 7.5];%目标风味
% FlavorWeight = [5; 1; 1; 1; 1; 1];
FlavorWeight = [0.1, 0.2, 0.2, 0.2, 0.2, 0.2]';%风味权重
AAPrice = [128; 95; 50; 100; 75; 53; 50; 57; 81];%成本

NumBest = 50;%输出最佳50组
NumInputs = 9;%9种输入氨基酸
NumFlavors = 6;%6种风味c
ErrorMax = 0.5;%最大误差
CostMax = 100;%最大成本
NumFile = 10 ;
DataBest = zeros(NumFile, NumBest, NumInputs);
FlavorBest = zeros(NumFile, NumBest, NumFlavors);
ScoreBest = zeros(NumFile, NumBest);
CostBest = zeros(NumFile, NumBest);
ErrorBest = zeros(NumFile, NumBest);
WeightError = 0.9;%误差权重
WeightCost = 0.1;%成本权重
%% Load Data
    for j = 0 : 8
        load(strcat('C:/Users/Admin/Desktop/MISOresult/MISO-Test-', num2str(j) , '-2022-10-18-19-56-32'))
        load(strcat('C:/Users/Admin/Desktop/TestData/Test-20-100-5-10-' , num2str(j+1) ,'-20221018T195531'))
        if j == 0  % getTable
         %加载九个文件   
            
            NumSamples = size(Score, 1);
%             NumFlavors = size(Score, 2);
%             NumInputs = size(TestData, 2);
            FlavorAll = zeros(NumSamples*9, NumFlavors);
        end
        FlavorAll(NumSamples*j+1 : NumSamples*(j+1), :) = Score;
        DataAll(NumSamples*j+1 : NumSamples*(j+1), :) = TestData;
    end
    FlavorAll = FlavorAll + FlavorInitial;
       %% Calculate Error
    Error = zeros(NumSamples*9, NumFlavors);
    for j = 1 : NumFlavors
        Error(:, j) = FlavorAll(:, j) - repmat(FlavorTarget(j), NumSamples*9, 1);%误差
    end
    Error_MSE = abs(Error).^2 * FlavorWeight / sum(FlavorWeight);  % 均方error
    Cost = DataAll * AAPrice;  % cost 
        % Calculate the allowed data
    DataMEFrontSum = sum(DataAll, 2);
    i = 5:12;
    xSum = find(0.5 <= DataMEFrontSum);%找到添加量在1.2以内的组分
    %%
    [Error_MSE, x] = sort(Error_MSE);
    Error_MSE = Error_MSE(1 : NumBest * 100);
    Cost = Cost(x(1 : NumBest * 100), :);
    DataAll = DataAll(x(1 : NumBest * 100), :);
    FlavorAll = FlavorAll(x(1 : NumBest * 100), :);
    ScoreAll = 10 - sqrt(Error_MSE / ErrorMax) .* WeightError - exp(Cost / CostMax) .* WeightCost;
     
    [ScoreSort, x] = sort(ScoreAll,1, 'descend');
    ErrorSort = Error_MSE(x, :);
    CostSort = Cost(x, :);
    DataSort = DataAll(x, :);
    FlavorSort = FlavorAll(x, :);
    ScoreSort = unique(ScoreSort,'rows','stable');
    ErrorSort = unique(ErrorSort,'rows','stable');
    CostSort = unique(CostSort,'rows','stable');
    DataSort = unique(DataSort,'rows','stable');
    FlavorSort = unique(FlavorSort,'rows','stable');

   
%%

[a,b,c] = size(DataBest);
DataBest1 = reshape(DataBest,a*b,c);
[a,b,c] = size(FlavorBest);
FlavorBest = reshape(FlavorBest,a*b,c);%矩阵降维
DataSort = DataSort';
FlavorSort = FlavorSort';
ScoreSort = ScoreSort';
CostSort = CostSort';


Time = datestr(now, 30);%时间
save(strcat('C:/Users/Admin/Desktop/score/三号/Score-2-', Time),'DataSort', 'CostSort', 'FlavorSort', 'ScoreSort')%储存
