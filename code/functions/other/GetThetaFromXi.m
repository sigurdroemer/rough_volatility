function theta = GetThetaFromXi(v0,H,t,xi)
% Description: Consider (in the context of the rough Heston model) the following,
%
%   xi(t) = v0 + int_0^t K(t-s) theta(s) ds,	t >= 0,
%
% where v0 >= 0, K(t) = (1/gamma(H+1/2))*t^(H-1/2), H between 0 and 1/2, and theta 
% is a deterministic function that is piecewise constant betwen the time points
% 0 = t(0) < t(1) < t(2) < ... < t(n). 
%
% Given v0 and xi(t(i)), i=1,2,...,n, we return the theta-values that 
% ensure that t -> (t,xi(t)) goes through the points (t(i),xi(t(i)), i=1,2,...,n.
%
% Parameters:
%   v0:	    [1x1 real] Instantaneous variance.
%   H:      [1x1 real] Hurst exponent.
%   t:      [nx1 real] Time points t(1),t(2),...,t(n).
%   xi:     [nx1 real] Forward variances at the maturities t(1),t(2),...,t(n).
%   
% Output:
%   theta:  [nx1 real] Theta values.
%

t_ext = [0;t];
n = size(xi,1);
theta = zeros(n,1);
for i=1:n
	if i==1
		wii = (1./gamma(H+3/2))*t(i)^(H+1/2);
		wik_theta_sum = 0;
	else
		wik_theta_sum = 0;
		for j=2:i
			wik_theta_sum = wik_theta_sum + (1/gamma(H+3/2))*( (t(i)-t_ext(j-1)).^(H+1/2)...
                                           - (t(i)-t_ext(j)).^(H+1/2) ).*theta(j-1);
		end
		wii = (1/gamma(H+3/2))*(t(i) - t(i-1))^(H+1/2);
	end
	theta(i) = (xi(i) - v0 - wik_theta_sum) ./ wii;
	
end

end

