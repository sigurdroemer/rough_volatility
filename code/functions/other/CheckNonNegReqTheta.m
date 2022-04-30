function [valid, val] = CheckNonNegReqTheta(v0,H,t,theta)
% Description: In the context of the rough Heston model, we check if 
% theta(t)dt + v0*L(dt) is a non-negative measure where 
% L(dt) = t^(-H-1/2)/gamma(1/2-H) dt.
%
% Parameters:
%   v0:     [1x1 real]  Initial instantaneous variance.
%   H:      [1x1 real]  Hurst exponent.
%   t:      [Nx1 real]  Time points between which theta is piecewise constant 
% 						(zero excluded). We assume theta is extrapolated flat.
%   theta:  [Nx1 real]  Theta values for each interval.
%
% Output:
%   valid:  [1x1 logical] True if theta(t)dt + v0*L(dt) is a non-negative measure.
%   val:    [Nx1 real] The values that should be non-negative.

% Because the density of L(dt) = t^(-H-1/2)/gamma(1/2-H) dt is non-increasing, it 
% suffices to check the right end points:
val = theta + (v0/gamma(1/2-H))*t.^(-H-1/2);

% For the requirement to be satisfied for t -> infinity also, we need additionally:
val(end) = theta(end);

% Check if measure is non-negative:
valid = all(val >= 0);

end

