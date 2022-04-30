function val = elu(x,gradient)
% Description: Implements the exponential linear unit (ELU) activation
% function defined as
%
%   elu(x) = exp(x) - 1   for x <= 0
%   elu(x) = x            for x > 0
%
% Parameters:
%   x:        [nx1 real] Input values.
%   gradient: [1x1 logical] If true then we return the gradient (i.e. 
%             derivative), if false we just return the value. Default 
%             is false.
%
% Output:
%   val: [nx1 real] Value(s) or gradient(s).
%

    if ~gradient
        val = x;
        idxNonPos = x<=0;
        val(idxNonPos) = exp(x(idxNonPos)) - 1;
    else
       val = elu_gradient(x); 
    end
end

function val = elu_gradient(x)
% Description: Evaluates the gradient of the ELU activation function.
%
% Parameters:
%   x: [nx1 real] Input values.
%
% Output:
%   val: [nx1 real] Gradients.
%

    val = NaN(size(x));
    idxPos = x>0;
    val(~idxPos) = elu(x(~idxPos)) + 1;
    val(idxPos) = 1;
end