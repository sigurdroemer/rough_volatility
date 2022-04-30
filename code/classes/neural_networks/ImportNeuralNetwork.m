function nn = ImportNeuralNetwork(weightsfile,actFun)
% Description: Imports a neural network.
%
% Parameters:
%   weightsfile:  [1x1 string] Full path to a json file containing the
%                 weights, biases and input/output scalings. Let json{1},
%                 json{2},...,json{end} denote the fields of the json file.
%                 The fields json{1},json{2},...,json{end-5},json{end-4}
%                 must then contain the weight matrices and bias vectors
%                 of each layer. I.e. json{1} is the weight matrix of the
%                 first layer and json{2} the bias vector of the first
%                 layer, etc. Next the input/output scalings must come. 
%                 Specifically we scale an input vector as
%
%                       scaled input = (input - json{end-3}) / sqrt(json{end-2})
%
%                 and rescale the output vector as
%
%                       scaled output = output*sqrt(json{end}) + json{end-1}.
%
%   actFun:     [1x1 function] Vectorized activation function.
%
% Output:
%   nn: [1x1 NeuralNetwork] Object of class 'NeuralNetwork'.
%

    % Import basic network:
    jsonObj = jsondecode(fileread(weightsfile));
    nLayers = (size(jsonObj,1) - 4)/2;
    
    % Unpack weights and biases:
    [weights, biases] = deal(cell(nLayers,1));
    for i=1:nLayers
        weights{i} = jsonObj{2*i-1}';
        biases{i} = jsonObj{2*i};
    end
    
    % Import scalings:
    meanin = jsonObj{end-3}';
    varin = jsonObj{end-2}';
    meanout = jsonObj{end-1};
    varout = jsonObj{end};
    
    % Construct network:
    nn = NeuralNetwork(weights,biases,actFun,meanin,sqrt(varin),meanout,sqrt(varout));

end

