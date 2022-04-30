function c = bscall_3d(s0,K,r,T,sig2,q)
% Description: Implements the Black-Scholes call price formula but allows
% for some particular input dimensions (see below).
%
% Parameters:
%   s0:   [MxN real] Asset price.
%   K:    [1x1xL real] Strike price.
%   r:    [1x1 real] Interest rate.
%   T:    [1x1 real] Time-to-maturity.
%   sig2: [MxN real] Black-Scholes volatility squared.
%   q:    [1x1 real (optional)] Dividend yield (default is 0).
%
% Output:
%   c: [MxNxL real] Prices.
%

    if ~exist('q','var') || isempty(q); q = 0; end
    F = s0*exp((r-q)*T);
    totalVar = T.*sig2;
    totalStd = sqrt(totalVar);
    d1 = (log(s0) - log(K) + (r-q)*T) ./ totalStd + 0.5*totalStd;
    d2 = d1 - totalStd;
    c = exp(-r*T)*(F.*phi_cdf(d1) - K.*phi_cdf(d2));
    idxZeroVol = repmat(totalStd,1,1,size(K,3)) == 0;
    c_zerovol = subplus(s0.*exp(-q.*T) - K.*exp(-r.*T));
    c(idxZeroVol) = c_zerovol(idxZeroVol);
    
end

function phi = phi_cdf(z)
% Description: Implements the standard normal CDF. Code is extracted directly
% from matlabs 'normcdf'.Speed is improved moderately by bypassing the other
% code in 'normcdf'.
% 
% Parameters:
%   z: [MxNxL real] Input matrix.
%
% Output:
%   phi: [MxNxL real] Output matrix.
%

    phi = 0.5 * erfc(-z ./ sqrt(2));
    
end
