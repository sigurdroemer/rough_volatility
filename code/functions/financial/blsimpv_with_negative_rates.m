function iv = blsimpv_with_negative_rates(s0,K,r,T,q,priceObs,idxCall)
% Description: Matlabs 'blsimpv' function which computes implied volatilities 
% does not accept negative interest rates or dividend yields. This function 
% takes care of that.
%
% Parameters:
%   s0:         [1x1 real] Asset price.
%   K:          [MxN real] Strikes.
%   r:          [MxN real] Yields.
%   T:          [MxN real] Expirations.
%   q:          [MxN real] Dividend yields.
%   priceObs:   [MxN real] Observed option prices.
%   idxCall:    [MxN logical] True if observed price is for a call option,
%               false for a put option.
%   
% Output:
%   volatility: [MxN real] Implied volatilities.
%

% Transform the problem to forward prices:
priceObsAdj = priceObs.*exp(r.*T);
F = s0.*exp((r-q).*T);

% Call the blsimpv function:
iv = blsimpv(F,K,0,T,priceObsAdj,'Limit',5,'Yield',0,'Class',idxCall);

end

