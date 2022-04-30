function p = FourierPricingCall(s0,k,tau,phi,alpha,integration_function,...
                                 upper_bound,transform_domain,model,...
                                 throw_error_on_negative_time_value,...
                                 throw_error_on_integration_warning)
% Description: Implements Fourier pricing as in (Carr and Madan, 1999) and
% (Lord and Kahl, 2006) to return call option prices. We assume zero
% interest rate and dividends.
%
% Parameters:
%   s0:           [1x1 real] Initial asset price.
%   k:            [Nx1 real] Log-strikes.
%   tau:          [1x1 real] Time-to-expiration.
%   phi:          [1x1 function] The characteristic function of the log spot at 
%                 maturity. Must be vectorized.
%   alpha:        [1x1 real] Dampening parameter.
%   integration_function:       
%                 [1x1 function] Function that will integrate another
%                 function. Should take inputs as intFun(f,a,b) and then
%                 integrate f from a to b. If N > 1 this must allow
%                 integration of vector-valued functions.
%   upper_bound:  [1x1 real] Upper bound of integration domain, unless
%                 transfDomain = true in which case we integrate over
%                 [0,1] instead. 
%   transform_domain: 
%                 [1x1 logical] If true then we transform the integration
%                 domain to the interval [0,1]. Currently only possible
%                 for the Heston model.
%   model:        [1x1 PricingModelClass] Model to price under.
%   throw_error_on_negative_time_value:
%                 [1x1 logical] If true then we throw an error if a negative 
%                 time value is computed. If false we in that situation instead 
%                 show a warning and then set the time value to zero.
%   throw_error_on_integration_warning:
%                 [1x1 logical] If true then we throw an error if the numerical 
%                 integration throws a warning, if false we do not.
%
% Output:
%   p:  [NxM real] Option prices.
%
% References:
%   o Carr, P. and Madan, D. B., Option valuation using the fast Fourier 
%     transform. 1999, Journal of Computational Finance, 2(4), 61-73.
%   o Lord, R. and Kahl, C., Optimal Fourier inversion in semi-analytical 
%     option pricing. 2006, Tinbergen Institute Discussion Paper No. 2006-066/2.
%

    % Define integrand:
    psi = @(v) (  phi( v - (alpha + 1)*1i  ) ...
                 ./ ( alpha^2 + alpha - v.^2 + 1i*(2*alpha + 1).*v)  );
    integrand = @(v)(real(psi(v).*exp(-1i*v*k')).*(exp(-alpha*k')./pi));
    integrand = @(v)(integrand(v).');
    
    if transform_domain
        % We transform the integration domain to [0,1] (only developed with
        % the Heston model in mind).
        if ~strcmpi(class(model),'HestonClass')
            error(['FourierPricingCall: Transforming the domain of ',...
                   'integration is only valid if the model is of the type ', ...
                   '''HestonClass''.']);
        end
        rho = model.rho;omega = model.eta; v0 = model.v0; 
        kappa = model.kappa;theta = model.vbar;
        cinf = (sqrt(1-rho^2)/omega)*(v0 + kappa*theta*tau);
        integrand = @(x)(HestonTransfIntegrand(x,integrand,cinf));
    end
    
    % Remove last warning:
    lastwarn('');
    
    % Compute prices:
    if ~transform_domain
        p = integration_function(integrand,0,upper_bound);
    else
        % We transform the integration domain to [0,1]. 
        % Caution: Only implemented with Heston in mind.
        p = NaN(size(k));
        for i=1:size(p,1)
            if integrand(1) ~= 0
              % Integration sometimes work better when the integrand is 
              % normalised :
              p(i) = integrand(1).*integration_function(@(x)(integrand(x)./integrand(1)),0,1);
            else
              p(i) = integration_function(integrand,0,1);
            end
        end
    end
    
    % Check for a warning and throw an error if needed:
    warnMsg = lastwarn;
    if ~isempty(warnMsg) && throw_error_on_integration_warning
        error(warnMsg);
    end

    % Adjust prices depending on alpha, see e.g. equation (7) and (8) from
    % (Lord and Kahl, 2006):
    R = 0;K = exp(k);
    if alpha <= 0
        R = R + s0;
    end
    if alpha <= -1
        R = R - K;
    end
    if alpha==0
        R = R - 0.5*s0;
    end
    if alpha==-1
        R = R + 0.5*K;
    end
    p = p + R;
    
    % Check for negative time values:
    idxITM = K < s0;
    timeval = p;
    timeval(idxITM) = p(idxITM) - s0 + K(idxITM);
    idxTimeValNeg = timeval < 0;
    if any(idxTimeValNeg)
        if ~throw_error_on_negative_time_value
            warning(['FourierPricingCall: Negative time values computed.',...
                     ' Setting negative time values to zero.']);
        else
            error('FourierPricingCall: Negative time values computed.');
        end
        % Set negative time values to zero:
        p(idxTimeValNeg & idxITM) = s0 - K(idxTimeValNeg & idxITM);
        p(idxTimeValNeg & ~idxITM) = 0;
    end
    
end

function val = HestonTransfIntegrand(x,untransfIntegrand,cinf)
    idxZero = x==0;
    val = untransfIntegrand(-log(x)./cinf)./(cinf.*x.');
    val(idxZero) = 0;    
end

