classdef HestonClass < PricingModelClass
% Description: Implements the Heston model of (Heston, 1993).
%
% The model is briefly explained. Letting r(t) and delta(t) denote the  
% deterministic risk-free interest rate and continuous proportional dividend 
% yield (respectively) the asset price S(t) follows the dynamics
%
% dS(t) = S(t)(r(t) - delta(t))dt + S(t)sqrt(V(t))dW_2(t)
%
% under the risk-neutral measure. Here W_2(t) is a Brownian motion and V(t)
% the instantaneous variance process. The V(t) process is then modelled as
%
% dV(t) = kappa*(vbar - V(t))dt + eta*sqrt(V(t))dW_1(t)
%
% with W_1 another Brownian motion s.t. dW_1(t)dW_2(t) = rhodt for a 
% -1 <= rho <= 1 and where V(0) = v0 with v0 > 0. We also have 
% kappa,vbar,eta > 0.
%
% Properties:
%   o kappa:  [1x1 real] Mean reversion speed of V(t).
%   o vbar:   [1x1 real] Long term level of V(t).
%   o v0:     [1x1 real] Initial level of V(t).
%   o eta:    [1x1 real] Volatility-of-volatility parameter.
%   o rho:    [1x1 real] Correlation parameter.
%
% More properties are inherited from the PricingModelClass. The most important 
% being the obj.pricerSettings property where the settings for the pricing 
% algorithm are set. You should consult the PricingModelClass for an 
% explanation of how that object is to be interpreted. Below we only very 
% briefly explain what settings are possible for the Heston model:
%
% The obj.pricerSettings must be of the FourierPricerSettingsClass type
% as only pricing with fourier methods is implemented. Other than that there are
% no restrictions on what settings can be chosen among those explained in
% the description of that class.
%
% References:
%   o Heston, S. L., A closed-form solution for options with stochastic 
%     volatility with applications to bond and currency options. 1993, Review 
%     of Financial Studies, 6(2), 327-343.
%
    
properties
    kappa
    vbar
    v0
    eta
    rho
end
    
methods
   function obj = HestonClass(varargin)
    % Description: Constructor. 
    % 
    % Parameters: Inputs must be given in name-value pairs corresponding to the 
    % object properties. Note that some of these properties are inherited from 
    % the PricingModelClass.
    %
    % Note also that default values may be set if some properties are not
    % specified. 
    %
    % Output:
    %   [1x1 HestonClass] The object.
    %
    % Examples: 
    %   o model = HestonClass('kappa',2,'vbar',0.2^2,'eta',2,'rho',-0.6,...
    %                         'v0',0.2^2);
    % 
    
        obj.ParseConstructorInputs(varargin{:});

        % Set default price estimation settings:
        if isempty(obj.pricerSettings)
            intFun = @(f,a,b)(integral(f,a,b,'ArrayValued',true));
            obj.pricerSettings = FourierPricerSettingsClass('alpha',[],...
                   'transform_domain',false,'upper_bound_integration',Inf,...
                   'integration_function',intFun,...
                   'integration_function_allows_vector_valued',true,...
                   'throw_error_on_negative_time_value',false,...
                   'throw_error_on_integration_warning',false);
        end
        
        % Adjust the yield and dividend yield curve if needed:
        if isprop(obj,'y') && isnumeric(obj.y)
            obj.y = CurveClass('gridpoints',1,'values',obj.y);
        end        
        if isprop(obj,'q') && isnumeric(obj.q)
            obj.q = CurveClass('gridpoints',1,'values',obj.q);
        end        
        
   end      
   function [p, idxCall, stdErrors, iv] = GetPricesSub(obj,k,T,cartProd,retSE)
   % Description: Computes prices and/or implied volatilities of vanilla
   % options. This function is not intended to be used by an end-user but 
   % only as an auxiliary function to be used by the 'GetPrices' method of 
   % the PricingModelClass.
   %
   % Remark: 
   %    o Currently the outputted implied volatilities are empty. For the
   %      end-user these will be computed in the 'GetPrices' method of the
   %      PricingModelClass if requested.
   %
   % Parameters:
   %    k:        [nx1 real] Log-moneyness.
   %    T:        [mx1 real] Expirations.
   %    cartProd: [1x1 logical] If true then we return prices for 
   %              cartesian product of k and ttm vectors. Else we 
   %              assume n = m and return prices for each element.
   %    retSE:    [1x1 logical] If true we return standard errors,
   %              otherwise we don't. Default is false.
   %
   % Output: 
   %    p:       [nx1 or nxm real] Prices of either call or put options.
   %    idxCall: [nx1 or nxm logical] Entry is true if price is of a call 
   %             option, otherwise it is for a put.
   %    se:      [empty] Standard errors.
   %    iv:      [empty] Black-Scholes implied volatilities.
   %       
       
       if retSE
          error('HestonClass:GetPricesSub: Standard errors are not supported.');
       end

       if size(T,1) > 1 || ~cartProd
          error(['HestonClass:GetPricesSub: Function only supports expiry ',...
                  'inputs of size [1x1] and cartesian product must also be ',...
                  'set to true.']);
       end 

       [stdErrors, iv] = deal([]);

        p = FourierPricing(1,k,T,obj);
        idxCall = true(size(k,1),1);

   end
   function val = CharacteristicFun(obj,t,u)
   % Description: Implements the characteristic function of the log spot price 
   % here assuming zero interest rate and dividends and an initial asset price 
   % of 1.
   %
   % Specifically we return E[exp(i*u*log(S(t)))] where i is the imaginary unit 
   % and that under the assumption of zero drift and S(0) = 1.
   %
   % Parameters:
   %    t:  [1x1 real] See the description.
   %    u:  [Nx1 real] See the description.
   %
   % Output:
   %    val: [Nx1 complex] Value of the characteristic function.
   %
   % References:
   %    o Gatheral, J., The Volatility Surface: A Practitioner's Guide. 2006, 
   %      Wiley.
   %

       alpha = -0.5*u.*( u + 1i );
       beta = obj.kappa - obj.rho*obj.eta*1i.*u;
       gamm = obj.eta^2/2;
       d = sqrt(beta.^2 - 4*alpha*gamm);

       rp = (beta + d) / obj.eta^2;
       rm = (beta - d) / obj.eta^2;
       g = rm ./ rp;

       C = obj.kappa*(rm*t - (gamm^(-1))*log( (1 - g.*exp(-d*t)) ./ (1 - g) ));
       D = rm .* ( (1 - exp(-d*t)) ./ (1 - g.*exp(-d*t)) );

       val = exp(C*obj.vbar + D*obj.v0);

   end
   function [zeta_dm, zeta_dp] = MomentExplosionInitialGuess(obj)
    % Description: Returns valid initial guesses for computing moment
    % explosions. 
    %
    % Remark: The function uses lemma 3.1 from (Lord and Kahl, 2006) to find 
    % valid guesses for both the lower and upper moments.
    %
    % Output:
    %   zeta_dm:    [1x1 real] Initial guess of lower moment.
    %   zeta_p:     [1x1 real] Initial guess of upper moment.
    %
    % References: 
    %   o Lord, R. and Kahl, C., Optimal Fourier inversion in semi-analytical 
    %     option pricing. 2006, Tinbergen Institute Discussion Paper 
    %     No. 2006-066/2.
    %

        % Unpack variables in papers notation:
        omega = obj.eta; kappa = obj.kappa; rho = obj.rho;

        % Compute initial guesses:
        dummy1 = omega - 2*kappa*rho;
        dummy2 = 2*omega*(1-rho^2);
        dummy3 = (omega - 2*kappa*rho)^2 + 4*(1-rho^2)*kappa^2;
        zeta_dm = (dummy1 - sqrt(dummy3)) / dummy2;
        zeta_dp = (dummy1 + sqrt(dummy3))/dummy2;
        
   end
   function Tstar = GetExplosionTime(obj,omega)
   % Description: Finds the moment explosion time T* for a moment omega.
   % That is we return the value T* s.t. E[S(T)^(omega)] is finite for all
   % T < T* and infinite for all T >= T^*. Here S(t) denotes the asset price.
   %
   % Remark: The implementation follows proposition 3.1 from (Andersen and
   % Piterbarg, 2005).
   %
   % Parameters:
   %    omega:  [Nx1 real] Moments.
   %
   % Output:
   %    Tstar:  [Nx1 real] Explosion times.
   %
   % References: 
   %    o Andersen, L.B.G. and Piterbarg, V.V., Moment explosions in
   %      stochastic volatility models. 2005, 11(1), 29-50.
   %

      % Check assumptions:
      if any([obj.kappa,obj.vbar,obj.eta,obj.v0] <= 0)
            error('HestonClass: GetExplosionTime: Assumptions are not met.');
      end

      % Unpack model parameters (and change to the notation of the paper):
      lambda = 1; eps = obj.eta; kappa = obj.kappa;
      rho = obj.rho;

      % Compute the explosion times:
      k = 0.5*(lambda^2)*omega.*(omega - 1);
      b = 2*k/(eps^2);
      a = 2*(rho*eps*lambda*omega - kappa)/(eps^2);
      D = a.^2 - 4*b;

      if any(b <= 0) || any(k <= 0)
          error('HestonClass: GetExplosionTime: Assumptions are not met.');
      end

      Tstar = NaN(size(omega));
      idxCase_1 = D >= 0 & a < 0;
      idxCase_2 = D >= 0 & a > 0;
      idxCase_3 = D < 0;

      if any(idxCase_1)
          Tstar(idxCase_1) = Inf;
      end
      if any(idxCase_2)
          gamm = 0.5*sqrt(D(idxCase_2));
          Tstar(idxCase_2) = (gamm^(-1))*(eps^(-2))*log( (a(idxCase_2)./2 + gamm)...
              ./ (a(idxCase_2)./2 - gamm) );
      end
      if any(idxCase_3)
          beta = 0.5*sqrt(-D(idxCase_3));
          Tstar(idxCase_3) = 2*(beta.^(-1))*(eps^(-2)).*( pi*(a(idxCase_3)<0)...
              + atan(2*beta./a(idxCase_3)) );
      end

   end
end
end

