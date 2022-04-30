function p = bsput(s0,K,r,T,sig2,q)
% Description: Implements the Black-Scholes put price formula. 
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
%   p:  [Nx1 or NxM] Prices.
%

    if ~exist('q','var'); q = 0; end
    c = bscall(s0,K,r,T,sig2,q);
    p = c + K.*exp(-r.*T) - s0.*exp(-q.*T);
    
end