%% Initialize Script
clear;
serverRun = false;
project_folder = fileparts(fileparts(fileparts(matlab.desktop.editor.getActiveFilename)));
addpath(genpath(project_folder));
dataFolder = [project_folder,'\code\neural_networks\data'];

%% Define model
model = rHestonClass('H',0.1,'rho',-0.5,'nu',0.4,'v0',0.2^2);
model.y = CurveClass('gridpoints',1,'values',0);
model.q = CurveClass('gridpoints',1,'values',0);
T_theta = [(0.0025:0.0025:0.0175)';(0.02:0.02:0.14)';(0.16:0.12:1)';(1.25:0.25:2)';3];
model.theta = CurveClass('gridpoints',T_theta,'values',zeros(size(T_theta)));

% Define pricer settings:
model.charFunSettings.n = 200;
model.pricerSettings.throw_error_on_integration_warning = true;
model.pricerSettings.throw_error_on_negative_time_value = true;

%% Define inputs and outputs:
seed_train = 1435124;
seed_test = 5155122;

params = {'H','nu','rho','v0','theta'};

[H_settings,nu_settings,rho_settings,v0_settings,xi_settings] = deal(struct);

H_settings.method = 'unif_cont';
H_settings.lb = 0.00;
H_settings.ub = 0.50;

nu_settings.method = 'unif_cont';
nu_settings.lb = 0.10;
nu_settings.ub = 1.25;

rho_settings.method = 'unif_cont';
rho_settings.lb = -1.00;
rho_settings.ub =  0.00;

% The sampling of v0 will be overwritten by the sampling of xi used for both v0 and theta:
v0_settings.method = 'unif_cont';
v0_settings.lb = 0.05;
v0_settings.ub = 1;
v0_settings.post_sampling_transformation = @(x)(x.^2);

% Settings for sampling theta curves:
% * Note that the below specifies the sampling of forward variance curves and 
% that theta-curves (and v0) are then chosen to match these. See 'ComputeTrainingData'.
xi_settings.method = 'theta_special';
xi_settings.Ts = T_theta;
xi_settings.type = {'Heston';'Heston';'Heston';...
                    'Heston';'Heston';'Heston';...
                    'Heston';'Heston';'Heston';...
                    'Heston';'Heston';'Heston';... 
                    'piecewise_flat';'piecewise_flat';'piecewise_flat';
                    'flat';... 
                    'independent';... 
                    };
xi_settings.perc = [0.35*(4/10);0.35*(4/10);0.35*(2/10);...
                    0.05/3;0.05/3;0.05/3;...
                    0.05/3;0.05/3;0.05/3;...
                    0.25/3;0.25/3;0.25/3;...
                    0.15/3;0.15/3;0.15/3;...
                    0.05;
                    0.10...
                    ];

xi_settings.vol0_low = [0.05;0.05;0.05;...
                        0.05;0.05;0.05;...
                        0.30;0.30;0.30;...
                        0.05;0.05;0.05;...
                        0.05;0.05;0.05;...
                        0.05;...
                        0.05;...
                        ];
xi_settings.vol0_high = [1;1;1;...
                         0.30;0.30;0.30;...
                         1;1;1;...
                         0.30;0.30;0.30;...
                         1;1;1;...
                         0.30;...
                         1;...
                         ];
xi_settings.vs_rate_low = [0.05;0.05;0.05;...
                           0.30;0.30;0.30;...
                           0.05;0.05;0.05;...
                           0.05;0.05;0.05;...
                           0.05;0.05;0.05;...
                           0.05;...
                           0.05;...
                            ];
xi_settings.vs_rate_high = [1;1;1;...
                            1;1;1;...
                            0.3;0.3;0.3;...
                            0.30;0.30;0.30;...
                            1;1;1;...
                            0.30;...
                            1;...
                            ];
xi_settings.kappa_low = [0.01;0.01;0.01;...
                         20;20;20;...
                         20;20;20;...
                         0.01;0.01;0.01;...
                         NaN;NaN;NaN;...
                         NaN;...
                         NaN;...
                        ];
xi_settings.kappa_high = [20;20;20;...
                          100;100;100;...
                          100;100;100;...
                          20;20;20;...
                          NaN;NaN;NaN;...
                          NaN;...
                          NaN;...
                          ];
                      
% We set the i.i.d. noise lower than for the rough Bergomi models in order to avoid
% discarding too many samples in certain parts of the sampling space:
xi_settings.eps_noise = [0.025;0.05;0.2;...
                        0.025;0.05;0.2;...
                        0.025;0.05;0.2;...
                        0.025;0.05;0.2;...
                        0;0.025;0.05;...
                        0.025;...
                        0;...
                        ]./10;                     
                                       
xi_settings.num_flat_sections = [NaN;NaN;NaN;...
                                 NaN;NaN;NaN;...
                                 NaN;NaN;NaN;...
                                 NaN;NaN;NaN;...
                                 3;3;3;...
                                 NaN;...
                                 NaN];
xi_settings.lb = 0.05.^2;
xi_settings.ub = 1;

samplingSettings = {H_settings,nu_settings,rho_settings,v0_settings,xi_settings};

% Choose the number of parallel workers:
nWorkers = 0;

% Load contracts:
k = importdata([dataFolder,'\logMoneyness.txt']);
T = importdata([dataFolder,'\expiries.txt']);

% Set folders to save temporary results:
saveFolderTrain = [dataFolder,'\temp\rheston_train_temp'];
saveFolderTest = [dataFolder,'\temp\rheston_test_temp'];
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
                                        'getInputsOnly',false,...
                                        'seed',seed_train);                                 
                                    
% Test data:
rng(seed_test);
[testData,testHeader,testInputFail,testErrFail] = ComputeTrainingData(...
                                         model,24000,params,...
                                         samplingSettings,k,T,...
                                        'use_sobol',false,...
                                        'tmpFolder',saveFolderTest,...
                                        'saveEvery',40000,...
                                        'nWorkers',nWorkers,...
                                        'folderSlash',folderSlash,...
                                        'checkSmileConvexity',true,...
                                        'checkMertonsTunnel',true,...
                                        'getInputsOnly',false,...
                                        'seed',seed_test);

%% Convert (v0,theta) to xi:
% Training data:
for i=1:size(trainData,1)
    model.H = trainData(i,1);
    model.v0 = trainData(i,4);
    model.theta.values = trainData(i,5:31)';
    trainData(i,5:31) = model.GetXi0(T_theta)';
end

% Update header:
ii = 0;
for i=4:31
    trainHeader{i} = ['xi_', num2str(ii)];
    ii = ii + 1;
end

% Test data:
for i=1:size(testData,1)
    model.H = testData(i,1);
    model.v0 = testData(i,4);
    model.theta.values = testData(i,5:31)';
    testData(i,5:31) = model.GetXi0(T_theta)';
end

% Update header:
ii = 0;
for i=4:31
    testHeader{i} = ['xi_', num2str(ii)];
    ii = ii + 1;
end


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
idxPar1 = (1:4+max(find(T_theta < 0.008))+1)';
idxPar2 = (1:4+max(find(T_theta < 0.03))+1)';
idxPar3 = (1:4+max(find(T_theta < 0.12))+1)';
idxPar4 = (1:4+max(find(T_theta < 0.40))+1)';
idxPar5 = (1:4+max(find(T_theta < 1.00))+1)';
idxPar6 = (1:4+max(find(T_theta < 3.00))+1)';

% Split dataset:
nParTotal = max(find(strcmpi(trainHeader','price_1'))) - 1;
idx1 = [idxPar1;nParTotal + find(T <= 0.008)];
idx2 = [idxPar2;nParTotal + find(T > 0.008 & T <= 0.03)];
idx3 = [idxPar3;nParTotal + find(T > 0.03 & T <= 0.12)];
idx4 = [idxPar4;nParTotal + find(T > 0.12 & T <= 0.40)];
idx5 = [idxPar5;nParTotal + find(T > 0.40 & T <= 1.00)];
idx6 = [idxPar6;nParTotal + find(T > 1.00)];

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

writetable(trainData_iv_1_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_training_data_1.csv']);
writetable(trainData_iv_2_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_training_data_2.csv']);
writetable(trainData_iv_3_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_training_data_3.csv']);
writetable(trainData_iv_4_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_training_data_4.csv']);
writetable(trainData_iv_5_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_training_data_5.csv']);
writetable(trainData_iv_6_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_training_data_6.csv']);

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

writetable(testData_iv_1_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_test_data_1.csv']);
writetable(testData_iv_2_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_test_data_2.csv']);
writetable(testData_iv_3_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_test_data_3.csv']);
writetable(testData_iv_4_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_test_data_4.csv']);
writetable(testData_iv_5_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_test_data_5.csv']);
writetable(testData_iv_6_tbl,[dataFolder,'\training_and_test_data\rheston\rheston_test_data_6.csv']);