function arb = CheckConvexity(c_vec,k_vec,T_vec,s0)
% Description: Checks call option prices for convexity.
%
% Assumptions: 
%   o Zero interest rate and dividends. 
%   o Assumes log-moneyness points are on an equidistant grid for each fixed 
%     expiry.
%
% Parameters: 
%   c_vec: [Nx1 real] Call option prices.
%   k_vec: [Nx1 real] Log-moneyness.
%   T_vec: [Nx1 real] Expiries.
%   s0:    [1x1 real] Spot price.
%
% Output:
%   arb: [1x1 logical] True if convexity failed otherwise false.
%

    uniqT = unique(T_vec);
    arb = false;
    for i=1:size(uniqT,1)
        idxSmile = T_vec == uniqT(i);
        [k, idxSort] = sort(k_vec(idxSmile));
        cSub = c_vec(idxSmile);
        cSub = cSub(idxSort);
        K = exp(k).*s0;

        dk = diff(k(1:2));

        fCenter = cSub(2:end-1);
        fUp = cSub(3:end);
        fDown = cSub(1:end-2);

        dcdk = (fUp - fDown) ./ (2*dk);
        dc2dk2 = (fUp - 2*fCenter + fDown) ./ (dk^2);
        dc2dK2 = (1./(K(2:end-1).^2)).*(dc2dk2 - dcdk);
        if any(dc2dK2 < 0)
            arb = true;
        end
    end

end

