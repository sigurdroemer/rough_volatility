classdef RandomNumbersClass < GenericSuperClass
% Description: Class storing random numbers.
%
% Properties:
%   o numbers: [1x1 struct] Containing (in some way) the random numbers 
%              that are to be stored.
%    

properties
    numbers 
end

methods
    function obj = RandomNumbersClass(varargin)
    % Description: Constructor.
    %
    % Parameters: Inputs must be in name-value pairs corresponding to the
    % object properties. 
    %
        obj.ParseConstructorInputs(varargin{:});
    end
end

end

