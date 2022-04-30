classdef ImpliedVolatilitySurfaceClass < handle
% Description: This object holds an implied volatility surface.
%
% Properties:
%   s0: [1x1 real] Spot price.
%   K:  [NxM real] Strikes.
%   k:  [NxM real] Log-moneyness values, defined as log(strike/forward).
%   F:  [NxM real] Forward prices.
%   T:  [NxM real] Expiries.
%   iv: [NxM real] Implied volatilities.
%   y:  [1x1 CurveClass] Yield curve.
%   q:  [1x1 CurveClass] Dividend yield curve.
%   



properties
    s0
    K
    k
    F
    T
    iv
    y
    q
end

methods
    function obj = ImpliedVolatilitySurfaceClass(s0, k, T, iv, y, q)
    % Description: Constructor.
    %
    % Remark: Most object properties are as explained below. The others are
    % explained here:
    %   o K: [NxM real] Strikes.    
    %   o F: [NxM real] Forward prices.
    %
    % Parameters:
    % s0: [1x1 real] Spot price.  
    % k:  [NxM real] Log-moneyness values.        
    % T:  [NxM real] Expiries.
    % iv: [NxM real] Implied volatilities.
    % y:  [1x1 CurveClass] Yield curve.
    % q:  [1x1 CurveClass] Dividend yield curve.
    %
    % Output: [1x1 ImpliedVolatilitySurfaceClass] The object.
    %

        if any(isnan(iv(:)))
           warning('ImpliedVolatilitySurfaceClass:Constructor: NaNs detected!');
        end
        
        obj.s0 = s0;
        obj.k = k;
        obj.F = s0 .* exp((y.Eval(T)-q.Eval(T)).*T);
        obj.K = obj.F.*exp(k);
        obj.T = T;
        obj.iv = iv;
        obj.y = y.DeepCopy();
        obj.q = q.DeepCopy();
        
    end
end

end

