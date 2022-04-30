function model = LoadHestonNeuralNetwork(projectFolder)
% Description: Loads the neural network based implementation of the Heston 
% model. Inputs should be given in the order: kappa, vbar, eta, rho, v0.
%
% Output:
%   model: [1x1 NeuralNetworkPricingModel] The neural network based pricing model.
%

    % Locate files:
    codeFolder = [projectFolder,'\code'];
    dataFolder = [codeFolder,'\neural_networks\data'];
    weightsFolder = [dataFolder,'\neural_network_weights\heston'];

    jsonFiles = {'heston_weights_1.json';...
                 'heston_weights_2.json';...
                 'heston_weights_3.json';...
                 'heston_weights_4.json';...
                 'heston_weights_5.json';...
                 'heston_weights_6.json'};

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
    lb = [0,0.05^2,0,-1,0.05^2]';
    ub = [25,1,10,0,1]';
    
    % Construct the neural network pricing model:
    model = NeuralNetworkPricingModel(networks,inputIdx,outputIdx,...
                                      k,T,lb,ub,'Heston');
end
