function c = bscall(s0,K,r,T,sig2,q)
% Description: Implements the Black-Scholes call price formula. 
%
% Parameters:
%   s0:     [Nx1 or 1x1 real] Asset price.
%   K:      [Nx1 or 1xM real] Strike price.
%   r:      [Nx1 or 1x1 real] Interest rate.
%   T:      [Nx1 or 1x1 real] Time-to-maturity.
%   sig2:   [Nx1 real] Black-Scholes volatility squared.
%   q:      [Nx1 or 1x1 real (optional)] Dividend yield (default is 0).
%
% Output:
%   c:  [Nx1 or NxM] Prices.
%

    if ~exist('q','var'); q = 0; end
    F = s0.*exp((r-q).*T);
    totalVar = T.*sig2;
    totalStd = sqrt(totalVar);
    logs0 = log(s0);
    d1 = (logs0 - log(K) + (r-q).*T) ./ totalStd + 0.5*totalStd;
    d2 = d1 - totalStd;
    c = exp(-r.*T).*(F.*phi_cdf(d1) - K.*phi_cdf(d2));
    
    if any(any(totalStd==0))
        idxZeroVol = repmat(totalStd,1,size(K,2)) == 0;
        tmp = subplus(s0.*exp(-q.*T) - K.*exp(-r.*T));
        c(idxZeroVol) = tmp(idxZeroVol);
    end
    
end

function phi = phi_cdf(z) 
% Description: Implements the standard normal CDF. Code is extracted directly 
% from matlabs 'normcdf'. Speed is improved moderately by bypassing the other 
% code in 'normcdf'.
%
% Parameters:
%   z: [MxN real] Inputs values.
%
% Output:
%   phi: [MxN real] Output values.
%

    phi = 0.5 * erfc(-z ./ sqrt(2));
    
end
