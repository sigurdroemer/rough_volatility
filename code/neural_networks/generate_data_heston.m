%% Initialize Script
clear;
serverRun = false;
project_folder = fileparts(fileparts(fileparts(matlab.desktop.editor.getActiveFilename)));
addpath(genpath(project_folder));
dataFolder = [project_folder,'\code\neural_networks\data'];


%% Define model
s0=100;r=CurveClass('gridpoints',1,'values',0);
q=CurveClass('gridpoints',1,'values',0);
kappa=2.2;eta=1.23;vbar=0.25^2;v0=0.25^2;rho=-0.80;
model = HestonClass('kappa',kappa,'vbar',vbar,'eta',eta,'rho',rho,'v0',v0,...
                     's0',s0,'y',r,'q',q);
model.pricerSettings.throw_error_on_negative_time_value = true;
model.pricerSettings.throw_error_on_integration_warning = true;

%% Define inputs and outputs:
seed_train = 1435127;
seed_test = 5155127;

params = {'kappa','vbar','eta','rho','v0'};

[kappa_settings,vbar_settings,eta_settings,rho_settings,v0_settings] = deal(struct);

kappa_settings.method = 'unif_cont';
kappa_settings.lb = 0.00;
kappa_settings.ub = 25;

vbar_settings.method = 'unif_cont';
vbar_settings.lb = 0.05;
vbar_settings.ub = 1;
vbar_settings.post_sampling_transformation = @(x)(x.^2);

eta_settings.method = 'unif_cont';
eta_settings.lb = 0.0;
eta_settings.ub = 10;

rho_settings.method = 'unif_cont';
rho_settings.lb = -1.00;
rho_settings.ub =  0.00;

v0_settings.method = 'unif_cont';
v0_settings.lb = 0.05;
v0_settings.ub = 1;
v0_settings.post_sampling_transformation = @(x)(x.^2);

samplingSettings = {kappa_settings,vbar_settings,eta_settings,rho_settings,v0_settings};

% Choose the number of parallel workers:
nWorkers = 0;

% Load contracts:
k = importdata([dataFolder,'\logMoneyness.txt']);
T = importdata([dataFolder,'\expiries.txt']);

% Set folders to save temporary results:
saveFolderTrain = [dataFolder,'\temp\heston_train_temp'];
saveFolderTest = [dataFolder,'\temp\heston_test_temp'];
folderSlash = '\';

%% Collect sample data:
% Training data:
rng(seed_train);
[trainData,trainHeader,trainInputFail,trainErrFail] = ComputeTrainingData(...
                                                     model,136000,params,...
                                                     samplingSettings,k,T,...
                                                    'use_sobol',true,...
                                                    'tmpFolder',saveFolderTrain,...
                                                    'saveEvery',40000,...
                                                    'nWorkers',nWorkers,...
                                                    'folderSlash',folderSlash,...
                                                    'checkSmileConvexity',true,...
                                                    'checkMertonsTunnel',true,...
                                                    'getInputsOnly',false);

% Test data:
rng(seed_test);
[testData,testHeader,testInputFail,testErrFail] = ComputeTrainingData(...
                                         model,24000,params,...
                                         samplingSettings,k,T,...
                                        'use_sobol',false,...
                                        'tmpFolder',saveFolderTest,...
                                        'saveEvery',24000,...
                                        'nWorkers',nWorkers,...
                                        'folderSlash',folderSlash,...
                                        'checkSmileConvexity',true,...
                                        'checkMertonsTunnel',true,...
                                        'getInputsOnly',false);
                                    


%% Convert to implied volatility
% Training data:
[trainData_iv, idxNotValid] = ConvertTrainingDataToIV(trainData,trainHeader,...
                                                          model.s0,k,T,false);
if any(idxNotValid);error('Non-valid implied vol''s detected!');end
sum(idxNotValid)

% Test data:
[testData_iv, idxNotValid] = ConvertTrainingDataToIV(testData,testHeader,...
                                                          model.s0,k,T,false);
if any(idxNotValid);error('Non-valid implied vol''s detected!');end
sum(idxNotValid)

%save([dataFolder,'\training_and_test_data\','rheston_trainData.mat'],'trainData','trainInputFail','trainErrFail');
%save([dataFolder,'\training_and_test_data\','rheston_testData.mat'],'testData','testInputFail','testErrFail');


%% Split each dataset into three separate ones depending on the expiries:
idxPar = (1:5)';

% Split dataset:
nParTotal = max(find(strcmpi(trainHeader','price_1'))) - 1;
idx1 = [idxPar;nParTotal + find(T <= 0.008)];
idx2 = [idxPar;nParTotal + find(T > 0.008 & T <= 0.03)];
idx3 = [idxPar;nParTotal + find(T > 0.03 & T <= 0.12)];
idx4 = [idxPar;nParTotal + find(T > 0.12 & T <= 0.40)];
idx5 = [idxPar;nParTotal + find(T > 0.40 & T <= 1.00)];
idx6 = [idxPar;nParTotal + find(T > 1.00)];

trainData_iv_1 = trainData_iv(:,idx1);
trainData_iv_2 = trainData_iv(:,idx2);
trainData_iv_3 = trainData_iv(:,idx3);
trainData_iv_4 = trainData_iv(:,idx4);
trainData_iv_5 = trainData_iv(:,idx5);
trainData_iv_6 = trainData_iv(:,idx6);

header_1 = strrep(trainHeader(idx1),'price','iv');
header_2 = strrep(trainHeader(idx2),'price','iv');
header_3 = strrep(trainHeader(idx3),'price','iv');
header_4 = strrep(trainHeader(idx4),'price','iv');
header_5 = strrep(trainHeader(idx5),'price','iv');
header_6 = strrep(trainHeader(idx6),'price','iv');

% Convert to tables: 
trainData_iv_1_tbl = array2table(trainData_iv_1);
trainData_iv_2_tbl = array2table(trainData_iv_2);
trainData_iv_3_tbl = array2table(trainData_iv_3);
trainData_iv_4_tbl = array2table(trainData_iv_4);
trainData_iv_5_tbl = array2table(trainData_iv_5);
trainData_iv_6_tbl = array2table(trainData_iv_6);

trainData_iv_1_tbl.Properties.VariableNames = header_1;
trainData_iv_2_tbl.Properties.VariableNames = header_2;
trainData_iv_3_tbl.Properties.VariableNames = header_3;
trainData_iv_4_tbl.Properties.VariableNames = header_4;
trainData_iv_5_tbl.Properties.VariableNames = header_5;
trainData_iv_6_tbl.Properties.VariableNames = header_6;

clear trainData_iv_1 trainData_iv_2 trainData_iv_3 trainData_iv_4 trainData_iv_5 trainData_iv_6 ...
      trainData_iv  trainData

writetable(trainData_iv_1_tbl,[dataFolder,'\training_and_test_data\heston\heston_training_data_1.csv']);
writetable(trainData_iv_2_tbl,[dataFolder,'\training_and_test_data\heston\heston_training_data_2.csv']);
writetable(trainData_iv_3_tbl,[dataFolder,'\training_and_test_data\heston\heston_training_data_3.csv']);
writetable(trainData_iv_4_tbl,[dataFolder,'\training_and_test_data\heston\heston_training_data_4.csv']);
writetable(trainData_iv_5_tbl,[dataFolder,'\training_and_test_data\heston\heston_training_data_5.csv']);
writetable(trainData_iv_6_tbl,[dataFolder,'\training_and_test_data\heston\heston_training_data_6.csv']);

clear trainData_iv_1_tbl trainData_iv_2_tbl trainData_iv_3_tbl ...
      trainData_iv_4_tbl trainData_iv_5_tbl trainData_iv_6_tbl

testData_iv_1 = testData_iv(:,idx1);
testData_iv_2 = testData_iv(:,idx2);
testData_iv_3 = testData_iv(:,idx3);
testData_iv_4 = testData_iv(:,idx4);
testData_iv_5 = testData_iv(:,idx5);
testData_iv_6 = testData_iv(:,idx6);

% Convert to tables: 
testData_iv_1_tbl = array2table(testData_iv_1);
testData_iv_2_tbl = array2table(testData_iv_2);
testData_iv_3_tbl = array2table(testData_iv_3);
testData_iv_4_tbl = array2table(testData_iv_4);
testData_iv_5_tbl = array2table(testData_iv_5);
testData_iv_6_tbl = array2table(testData_iv_6);

testData_iv_1_tbl.Properties.VariableNames = header_1;
testData_iv_2_tbl.Properties.VariableNames = header_2;
testData_iv_3_tbl.Properties.VariableNames = header_3;
testData_iv_4_tbl.Properties.VariableNames = header_4;
testData_iv_5_tbl.Properties.VariableNames = header_5;
testData_iv_6_tbl.Properties.VariableNames = header_6;

clear testData_iv_1 testData_iv_2 testData_iv_3 ...
      testData_iv_4 testData_iv_5 testData_iv_6 testData_iv testData

writetable(testData_iv_1_tbl,[dataFolder,'\training_and_test_data\heston\heston_test_data_1.csv']);
writetable(testData_iv_2_tbl,[dataFolder,'\training_and_test_data\heston\heston_test_data_2.csv']);
writetable(testData_iv_3_tbl,[dataFolder,'\training_and_test_data\heston\heston_test_data_3.csv']);
writetable(testData_iv_4_tbl,[dataFolder,'\training_and_test_data\heston\heston_test_data_4.csv']);
writetable(testData_iv_5_tbl,[dataFolder,'\training_and_test_data\heston\heston_test_data_5.csv']);
writetable(testData_iv_6_tbl,[dataFolder,'\training_and_test_data\heston\heston_test_data_6.csv']);

