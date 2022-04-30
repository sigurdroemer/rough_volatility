function p = FourierPricing(idxCall,k,ttm,model)
% Description: Returns prices of call or put options using Fourier pricing
% techniques. The settings for the computations are set in the pricerSettings
% property of the input model.
%
% Parameters:
%   idxCall: [1x1 logical] If true we price call options, otherwise puts.
%   k:       [Nx1 real] Log-moneyness values.
%   ttm:     [Mx1 real] Expiries.
%   model:   [1x1 PricingModelClass] Model to price under.
%
% Output:
%   p:    [NxM real] Option prices.
%
% References:
%   o Carr, P. and Madan, D. B., Option valuation using the fast Fourier 
%     transform. 1999, Journal of Computational Finance, 2(4), 61-73.
%   o Lord, R. and Kahl, C., Optimal Fourier inversion in semi-analytical 
%     option pricing. 2006, Tinbergen Institute Discussion Paper No. 2006-066/2.
%
    
    settings = model.pricerSettings;
    throwErrOnNegTimeVal = settings.throw_error_on_negative_time_value;
    throwErrOnIntWarning = settings.throw_error_on_integration_warning;
    
    % Define K matrix: Each column is a vector of strikes for a fixed expiry:
    F = model.s0*exp((model.y.Eval(ttm)-model.q.Eval(ttm)).*ttm);
    Kadj = exp(k);
    K = Kadj*F';
    
    % Unpack pricing settings:
    if ~isempty(settings.alpha)
        alpha = settings.alpha;
        fixedAlpha = true;
    else
        fixedAlpha = false;
    end
    intFun = settings.integration_function;
    intFunAllowsVectorVal = settings.integration_function_allows_vector_valued;
    transfDomain = settings.transform_domain;
    upperBound = settings.upper_bound_integration;
    
    % Initialize output matrix:
    p = NaN(size(K));
    
    % Loop over each expiry and compute call prices under the assumption of
    % zero interest rate, zero dividends and an initial asset price of 1
    % and all that using adjusted-strikes:
    for j=1:size(ttm,1)

        % Define the characteristic function:
        if isa(model,'rHestonClass')
            % Pre-evaluate the forward variance curve before defining the 
            % characteristic function (for performance):
            xiVals = model.CharacteristicFun(ttm(j),1,'only_final_timepoint',...
                                             [],true);
            phi = @(u) (model.CharacteristicFun(ttm(j),u,...
                                                'only_final_timepoint',...
                                                xiVals,false));
        else
            phi = @(u) (model.CharacteristicFun(ttm(j),u));
        end

        % Find optimal dampening parameters if needed:
        if ~fixedAlpha
            alpha = model.GetOptimalFourierAlpha(Kadj(:,j),ttm(j));    
        end

        if fixedAlpha && intFunAllowsVectorVal
            % Compute all strikes simultaneously:
            p(:,j) = FourierPricingCall(1,log(Kadj(:,j)),ttm(j),phi,alpha,...
                                       intFun,upperBound,transfDomain,...
                                       model,throwErrOnNegTimeVal,...
                                       throwErrOnIntWarning);
        else
            % Loop over strikes:
            for i=1:size(Kadj,1)
                if fixedAlpha;alphaUse=alpha;else alphaUse=alpha(i);end
                p(i,j) = FourierPricingCall(1,log(Kadj(i,j)),ttm(j),...
                                            phi,alphaUse,intFun,...
                                            upperBound,transfDomain,model,...
                                            throwErrOnNegTimeVal,...
                                            throwErrOnIntWarning);
            end
        end
    end
    
    % Put-call parity:
    if idxCall == -1
        p = p + Kadj - 1;
    end
    
    % Rescale:
    p = exp(-model.y.Eval(ttm)*ttm').*(F.').*p;
    
end