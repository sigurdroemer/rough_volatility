classdef NeuralNetwork
% Description: Implements a basic feed forward neural network.
%
% Properties:
%   weights:      [1xL cell] Cell array of weight matrices for each network layer.
%   biases:       [1xL cell] Cell array of bias vectors for each network layer.
%   actFun:       [1x1 function] Activation function (assumed to be the same for 
%                 each layer). Must be vectorized.
%   scaleMeanIn:  [Nx1 real] See 'scaleStdIn'.
%   scaleStdIn:   [Nx1 real] Before evaluating the network the inputs are scaled as 
%                 (input - scaleMeanIn) / scaleStdIn.
%   scaleMeanOut: [Mx1 real] See 'scaleStdOut'.
%   scaleStdOut:  [Mx1 real] Before returning the output of the neural network 
%                 the output is rescaled as output * scaleStdOut + scaleMeanOut.
%

properties
    weights
    biases
    actFun
    scaleMeanIn
    scaleStdIn
    scaleMeanOut
    scaleStdOut
end

methods
function obj = NeuralNetwork(weights,biases,actFun,scaleMeanIn,scaleStdIn,...
                             scaleMeanOut,scaleStdOut)
% Description: Constructor.
%
% Parameters: Exactly corresponds to the object properties. See the class
% description for more information.
% 
% Output:
%   obj: [1x1 NeuralNetwork] The neural network.
%

    obj.weights = weights;
    obj.biases = biases;
    obj.actFun = actFun;
    obj.scaleMeanIn = scaleMeanIn;
    obj.scaleMeanOut = scaleMeanOut;
    obj.scaleStdIn = scaleStdIn;
    obj.scaleStdOut = scaleStdOut;
    
end

function val = Eval(obj, input)
% Description: Evaluates the neural network given an input.
%
% Parameters:
%   input:      [Nx1 real] Input vector.
%
% Output:
%   val:        [Mx1 real] Output vector.
%

    % Evaluate network:
    val = (input - obj.scaleMeanIn)./obj.scaleStdIn;
    for i=1:size(obj.weights,1)-1
        val = obj.actFun(obj.weights{i}*val + obj.biases{i});
    end
    val = obj.scaleStdOut.*(obj.weights{end}*val ...
          + obj.biases{end}) + obj.scaleMeanOut;
      
end

end

end

