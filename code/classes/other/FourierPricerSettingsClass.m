classdef FourierPricerSettingsClass < GenericSuperClass
% Description: Class to hold settings for a pricer using numerical integration 
% based on the Fourier transform.
%
% Properties:
%   o alpha:                    [1x1 real or empty] The dampening parameter 
%                               from (Carr and Madan, 2001). Can be set to a 
%                               fixed [1x1 real] value in which case that will 
%                               be used. Can also be left empty in which case an 
%                               optimal alpha is computed using the techniques 
%                               from (Lord and Kahl, 2006).
%
%   o transform_domain:         [1x1 logical] If set to true the domain of 
%                               integration will be transformed from 
%                               [0,infinity) to [0,1].
%
%   o upper_bound_integration:  [1x1 real] While the integral to evaluate in 
%                               theory runs from [0,infinity) it will in 
%                               practise be truncated to 
%                               [0,upper_bound_integration]. That is unless 
%                               the transform_domain property is set to true in 
%                               which case the upper_bound_integration property 
%                               is left unused.
%
%   o integration_function:     [1x1 function] Integration function. This 
%                               function must take the inputs (f,a,b) where f 
%                               is a function and a < b the integration 
%                               endpoints. The intFun function should then 
%                               output the integral of f from a to b.
%
%   o integration_function_allows_vector_valued:
%                               [1x1 logical] Set to true if the
%                               integration_function property supports
%                               integration of vector-valued functions.
%                               Otherwise set it to false.
%
%   o throw_error_on_negative_time_value: 
%                               [1x1 logical] If set to true an error will be 
%                               thrown whenever a negative time value is 
%                               computed. If set to false and a negative
%                               time value is computed, a warning will be
%                               given and the time value will be set to zero.
%
%   o throw_error_on_integration_warning:
%                               [1x1 logical] If set to true an error will
%                               be thrown if numerical integration throws a
%                               warning. If false no error is thrown in
%                               that case.
%
% Remark: There may be additional restrictions on what property values are 
% allowed for a given model. 
%
% References:
%   o Carr, P. and Madan, D.B., Option valuation using the fast Fourier 
%     transform. 1999, Journal of Computational Finance, 2(4), 61-73.
%   o Lord, R. and Kahl, C., Optimal Fourier inversion in semi-analytical 
%     option pricing. 2006, Tinbergen Institute Discussion Paper No. 2006-066/2.
%
    
properties
    alpha
    transform_domain
    upper_bound_integration
    integration_function
    integration_function_allows_vector_valued
    throw_error_on_negative_time_value
    throw_error_on_integration_warning
end
    
methods
    function obj = FourierPricerSettingsClass(varargin)
    % Description: Constructor. 
    % 
    % Parameters: Inputs must be given in name-value pairs corresponding to 
    % the object properties.
    %
    % Output:
    %   [1x1 FourierPricerSettingsClass] The object.
    %

        obj.ParseConstructorInputs(varargin{:});

    end
end

end

