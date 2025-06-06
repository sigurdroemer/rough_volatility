function [model, Txi] = LoadrHestonNeuralNetwork(projectFolder)
% Description: Loads the neural network based implementation of the rough Heston 
% model. Inputs should be given in the order: H, nu, rho, xi0, xi1, xi2,..., xi27.
%
% Output:
%   model: [1x1 NeuralNetworkPricingModel] The neural network based pricing model.
%   Txi:   [27x1 real] Maturities of the forward variances.
%

    % Locate files:
    codeFolder = [projectFolder,'\code'];
    dataFolder = [codeFolder,'\neural_networks\data'];
    weightsFolder = [dataFolder,'\neural_network_weights\rheston'];

    jsonFiles = {'rheston_weights_1.json';...
                 'rheston_weights_2.json';...
                 'rheston_weights_3.json';...
                 'rheston_weights_4.json';...
                 'rheston_weights_5.json';...
                 'rheston_weights_6.json'};

	% Load expiries and moneyness values:
    k = importdata([dataFolder,'\logMoneyness.txt']);
    T = importdata([dataFolder,'\expiries.txt']);
    
    % Load neural networks
    [networks,inputIdx,outputIdx] = deal(cell(size(jsonFiles)));
    idxOutStart = 1;
    for i=1:size(jsonFiles,1)
        networks{i} = ImportNeuralNetwork([weightsFolder,'\',jsonFiles{i}],@(x)(elu(x,false)));
        inputIdx{i} = (1:size(networks{i}.scaleMeanIn,1))';
        idxOutEnd = idxOutStart + size(networks{i}.scaleMeanOut,1) - 1;
        outputIdx{i} = (idxOutStart:idxOutEnd)';
        idxOutStart = idxOutEnd + 1;
    end
    
    % Set the forward variance grid points and the input bounds:
    Txi = [0;(0.0025:0.0025:0.0175)';(0.02:0.02:0.14)';(0.16:0.12:1)';(1.25:0.25:2)';3];
    lb = [0.0,0.10,-1,(0.05^2)*ones(1,size(Txi,1))]';
    ub = [0.50,1.25,0,ones(1,size(Txi,1))]';
    
    % Construct the neural network pricing model:
    model = NeuralNetworkPricingModel(networks,inputIdx,outputIdx,...
                                      k,T,lb,ub,'rHeston');
end
