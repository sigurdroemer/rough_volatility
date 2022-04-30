%% Clear workspace and load code:
clear;
project_folder = fileparts(fileparts(fileparts(matlab.desktop.editor.getActiveFilename)));
addpath(genpath(project_folder));

%% Load the extended rough Bergomi neural network(s):
[model, Txi] = LoadrBergomiExtendedNeuralNetwork(project_folder);

% Inspect the model object:
model

% The vector below shows the grid points of the piecewise flat forward variance curve:
Txi

%% Plot the contract grid:
figure;
scatter(model.T,model.k,'.');
xlabel('Expiration');ylabel('Log-moneyness');
title('Contract grid');

%% Plot an example:
% Define the contracts:
k_obs = model.k;
T_obs = model.T;
cartProd = false;

% Choose the model parameters:
alpha = -0.45;
beta = -0.35;
eta = 2.1;
rho = -0.90;
xi = 0.30.^2;
par = [eta;rho;alpha;beta;repmat(xi,27,1)];

% Evaluate the neural network:
tic;
[iv, k_obs, T_obs] = model.Eval(par,k_obs,T_obs,cartProd);
toc;

% Compute true prices:
model_true = rBergomiExtClass('alpha',alpha,'beta',beta,'eta',eta,'rho',rho,'xi',xi);
model_true.pricerSettings.N = 25000; % number of paths to simulate
[iv_true,~,~,~,se] = model_true.GetPrices(k_obs,T_obs,false,'priceType','implied_volatility',...
                                              'standardErrors',true);

                                          
% Plot:
uniqT = unique(T_obs);
idxPlot = [9,22,39,64];
figure;
for i=1:size(idxPlot,2)
    subplot(2,2,i);
    idx = T_obs == uniqT(idxPlot(i));
    plot(k_obs(idx),iv_true(idx),'-','Color','blue','DisplayName','Actual','LineWidth',2);hold on;
    plot(k_obs(idx),iv(idx),'--','Color','red','DisplayName','Neural network','LineWidth',2);
    plot(k_obs(idx),iv_true(idx)-se(idx)*1.96,'--','Color','blue','LineWidth',0.7);hold on;
    plot(k_obs(idx),iv_true(idx)+se(idx)*1.96,'--','Color','blue','LineWidth',0.7);hold on;
    xlabel('Log-moneyness');
    ylabel('Implied volatility');
    title(['T = ',num2str(uniqT(idxPlot(i)))]);
    legend('Monte Carlo','Neural network','95 % Confidence interval (Monte Carlo)')
end


%% Load and filter example contracts:
% Load example contracts:
tmp = importdata([project_folder,'\get_started\example_contracts.txt']);
k_orig = tmp(:,1);
T_orig = tmp(:,2);

figure;
plot(T_orig,k_orig,'.');
title('Observed contracts'); 
xlabel('Expiry');
ylabel('Log-moneyness');
hold on;

% Filter contracts:
[k_obs,T_obs] = model.FilterContracts(k_orig,T_orig,false);
plot(T_obs,k_obs,'.');
legend('Original observed contracts','Those within the supported domain.');

%% How to speed up (repeated) evaluation:
% When repeatedly computing prices over the same contracts (although over
% possibly different model parameters) it is possible to perform a number
% of pre-computations to speed things up. This is very useful when
% calibrating. We illustrate this below.

nRuns = 1000;

% Without pre-computations:
tic
for i=1:nRuns
    iv = model.Eval(par,k_obs,T_obs,false,[]);
end
toc;

% With pre-computations:
% Important remark: Contracts must be sorted appropriately 
% first, see also the description of the PerformPrecomputations 
% method of the NeuralNetworkPricingModel class:
tmp = sortrows([T_obs,k_obs]);
T_obs = tmp(:,1);
k_obs = tmp(:,2);
preCompInfo = model.PerformPrecomputations(k_obs,T_obs);
tic
for i=1:nRuns
    iv = model.Eval(par,k_obs,T_obs,false,preCompInfo);
end
toc;

% It is also possible to skip a number of validation checks and 
% computations in the evaluation function if you can ensure certain 
% assumptions on the inputs are met, e.g. contracts should be sorted 
% etc. One should use this option with care - please read the 
% description for the Eval method of the NeuralNetworkPricingModel 
% class for the exact assumptions on the inputs.

%Example:
skipChecks = true;
tmp = sortrows([T_obs,k_obs]);
T_obs = tmp(:,1);
k_obs = tmp(:,2);

% With pre-computations and without checks:
tic
for i=1:nRuns
    iv = model.Eval(par,k_obs,T_obs,false,preCompInfo,skipChecks);
end
toc;

%% Calibration example:
% Set parameters:
alpha = -0.45;
beta = -0.25;
eta = 2.1;
rho = -0.90;
xi = 0.15.^2;
par_true = [eta;rho;alpha;beta;repmat(xi,27,1)];

% Filter contracts to the neural network domain:
[k_obs,T_obs] = model.FilterContracts(k_orig,T_orig,false);

% Sort contracts:
tmp = sortrows([T_obs,k_obs]);
T_obs = tmp(:,1);
k_obs = tmp(:,2);

% Perform pre-computations (on sorted contracts):
preCompInfo = model.PerformPrecomputations(k_obs,T_obs);

% Generate synthetic 'observed' prices:
iv_obs = model.Eval(par_true,k_obs,T_obs,false,preCompInfo);

% Set optimizer settings:
options = optimoptions('lsqnonlin','Algorithm','trust-region-reflective',...
'MaxFunctionEvaluations',10^4,'MaxIterations',10^4,'FunctionTolerance',10^(-12),...
'StepTolerance',10^(-4),'FiniteDifferenceStepSize',10^(-4),'Display','off');

% Merge sections of the forward variance curve:
uniqT = unique(T_obs);
grpPart = MergeForwardVarianceParameters(Txi,uniqT).';
idxConvParToNN = [1,2,3,4,grpPart + 4];

% Set the initial guess:
x0 = [2.5,-0.8,0,0,ones(1,max(grpPart))*0.3.^2].';

% Define the error function ('lsqnonlin' automatically squares and sums the result):
err = @(x)(iv_obs - model.Eval(x(idxConvParToNN),k_obs,T_obs,false,preCompInfo,true));

% Set parameter bounds:
lb = accumarray(idxConvParToNN',model.lb,[],@max);
ub = accumarray(idxConvParToNN',model.ub,[],@min);

% Run optimizer:
tic;
[xOpt,resnorm,residual,exitflag,output] = lsqnonlin(err,x0,lb,ub,options);
toc;

par_opt = xOpt(idxConvParToNN);

% Compare true and calibrated parameters:
[par_true, par_opt]

%% Illustrate fit:
% Pick an expiration:
T_plot = uniqT(4);

% Plot:
idx = T_obs == T_plot;
figure;
plot(k_obs(idx),iv_obs(idx),'x','DisplayName','Observed');
hold on;
dk = (max(k_obs(idx)) - min(k_obs(idx)))/100;
k_eval = (min(k_obs(idx)):dk:max(k_obs(idx)))';
iv_fit = model.Eval(par_opt,k_eval,T_plot);
plot(k_obs(idx),iv_obs(idx),'-','DisplayName','Calibrated model');
xlabel('Log-moneyness');ylabel('Implied volatility');
title(['Calibration fit (T = ', num2str(T_plot) ')']);
legend();



