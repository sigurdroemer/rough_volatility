function greekVal = BSGreek(greek,optType,s0,K,r,T,sig,q)
% Description: Implements the Black-Scholes greeks.
%
% Parameters: 
%   greek:   [1x1 string] Options are 'delta' and 'vega'.
%   optType: [1x1 string] Options are 'call' and 'put'. Can be left empty
%            if greek <> 'delta'.
%   s0:      [1x1 or NxM real] Asset price.
%   K:       [1x1 or NxM real] Strike.
%   r:       [1x1 or NxM real] Interest rate.
%   T:       [1x1 or NxM real] Time to maturity. Must be strictly positive.
%   sig:     [1x1 or NxM real] Volatility. Must be strictly positive.
%   q:       [1x1 or NxM real] Dividend yield.
%
% Output:
%   greekVal:   [1x1 or Nx1 real] Greek value.
%

    d1 = (log(s0./K) + T.*(r-q + sig.^2/2))./(sig.*sqrt(T));
    switch greek
        case 'delta'
            if strcmpi(optType,'call')
                greekVal = exp(-q.*T).*normcdf(d1);
            elseif strcmpi(optType,'put')
                greekVal = exp(-q.*T).*(normcdf(d1) - 1);
            end
        case 'vega'
            greekVal = s0.*exp(-q.*T).*normpdf(d1).*sqrt(T);
            idxZeroTotalVar = T.*sig == 0;
            greekVal(idxZeroTotalVar) = 0;
        otherwise 
            error(['BSGreek: Greek type ', greek, ' is not supported.']);
    end
end

