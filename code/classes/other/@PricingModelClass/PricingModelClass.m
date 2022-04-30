classdef PricingModelClass < GenericSuperClass
% Description: Superclass for pricing models.
%
% Properties:
%   o s0:               [1x1 real] Spot price.
%
%   o y:                [1x1 CurveClass] Interest rate yield curve. E.g. if
%                       r(t) is the deterministic risk-free interest rate
%                       then the yield curve is given by
%                           
%                           y(T) = (1/T) * int_0^T r(t) dt
%
%   o q:                [1x1 CurveClass] (integrated) dividend yield curve.
%                       E.g. if the underlying asset pays a continuous
%                       proportional (and deterministic) dividend of delta(t)
%                       then the (integrated) dividend yield curve is given by 
%                       
%                           q(T) = (1/T) * int_0^T delta(t) dt
%
%   o pricerSettings:   [1x1 FourierPricerSettingsClass or 
%                       1x1 MonteCarloPricerSettingsClass] Settings specifying 
%                       which pricing algorithm should be used and how. Which 
%                       ones are available and how the object should look will 
%                       depend on the particular model. 
%
    
properties
    s0
    y
    q
    pricerSettings
end
    
methods
function obj = PricingModelClass()
% Description: Constructor.
%
% Remark: If the yield curves are empty they default to flat zero curves.
% If the spot price is not set it defaults to a value of 100.
%
% Output:
%   [1x1 PricingModelClass] The object.
%

    if isempty(obj.y)
       obj.y = CurveClass('gridpoints',1,'values',0);
    end
    if isempty(obj.q)
       obj.q = CurveClass('gridpoints',1,'values',0);
    end
    if isempty(obj.s0)
        obj.s0 = 100;
    end

end
function [pricesOrIV,k_mat,ttm_mat,idxCall,se] = GetPrices(obj,k,ttm,...
                                                           cartProd,varargin)
% Description: Computes prices of put and call options.
%
% Main parameters:
%   k:        [Nx1 real] Log-moneyness values, defined as log(strike/forward).
%
%   ttm:      [Mx1 real] Expiries.
%
%   cartProd: [1x1 logical (optional)] If true then we return prices for 
%             the cartesian product of the log-moneyness and expiry vectors. 
%             Else we assume N = M and return one price for each row. If
%             this parameter is unspecified the default is to use a cartesian 
%             product if N <> M and otherwise not.
%
% Additional (optional) parameters that must be inputted in name-value pairs:
%   priceType:      [1x1 string] Format to return prices in. Options 
%                   are 'price', 'implied_volatility' and
%                   'implied_volatility_surface' (default).
%
%   optionType:     [1x1 string or Nx1 logical] The option type to return
%                   prices for. This parameter only has an effect if
%                   priceType = 'price'. The options for setting optionType to 
%                   a string are 'put', 'call'  and 'otm' (default). The latter 
%                   returns prices for out-of-the-money options. Moneyness is 
%                   here defined such that at-the-money is when strike = forward 
%                   price. Also, and assuming cartProd = false, then it is in 
%                   addition possible to input a Nx1 logical vector. An 
%                   entry in this vector should then be true to return the 
%                   price of a call option and false to return the price of a 
%                   put option.
%
%   standardErrors: [1x1 logical] If true we also compute standard errors. 
%                   Default is false.
%
%   use_existing_numbers: 
%                   [1x1 logical] Generally not recommended to be used by
%                   an end-user. In certain situations (and if one uses Monte
%                   Carlo estimation) random numbers (or even pre-simulated 
%                   stochastic process) needed for the computations can be 
%                   stored in the obj.pricerSettings.random property. By 
%                   setting this parameter to true the code will then attempt 
%                   to (re)use these existing numbers instead of overwriting 
%                   them by simulating new ones.
%
%   specialRun:     [1x1 logical] Not intended for use by an end-user.
%                   Needed for handling repeated calls to the function.
%                   Default is false.
%
% Output: 
%   pricesOrIV: [1x1 ImpliedVolatilitySurfaceClass or Nx1 real or NxM real] 
%               Prices or implied volatility in the requested format.
%
%   k_mat:      [Nx1 or NxM real] Log-moneyness values.
%
%   ttm_mat:    [Nx1 or NxM real] Expiries.
%
%   idxCall:    [Nx1 or NxM logical] An element is true if the corresponding 
%               price is for a call option and false if it is for a put option.
%
%   se:         [Nx1 real or NxM real] Standard errors. Empty if the 
%               standardErrors parameter is set to false.
%

   [pricesOrIV,k_mat,ttm_mat,se,idxCall] = deal([]);

   % Parse inputs:
   p = inputParser;
   exptPriceType = {'implied_volatility_surface','price','implied_volatility'}; 
   addParameter(p,'priceType','implied_volatility_surface',...
                @(x) any(validatestring(x,exptPriceType)));
   addParameter(p,'optionType','otm');
   addParameter(p,'standardErrors',false);
   addParameter(p,'use_existing_numbers',false);
   addParameter(p,'specialRun',false);
   parse(p,varargin{:});
   v2struct(p.Results);

   if ~exist('cartProd','var') || isempty(cartProd)
       if size(k,1) ~= size(ttm,1)
           cartProd = true;
       else
           cartProd = false;
       end
   end
   
   if ~exist('optionType','var') || isempty(optionType)
       optionType = 'otm';
   end
   
   if islogical(optionType)
       if cartProd || size(k,1) ~= size(optionType,1)
           error(['PricingModelClass:GetPrices: If the optionType ',...
                  'parameter is specified as a logical vector we must have ',...
                  'cartProd = false and optionType must then be of ',...
                  'the same length as the moneyness and expiry vectors.']);
       end
   else
       exptOptType = {'call','put','otm'};
       if ~any(validatestring(optionType,exptOptType))
           error(['PricingModelClass:GetPrices: If the parameter ',...
                  'optionType is specified as a string only the values ',...
                  '''call'', ''put'' and ''otm'' are allowed.']);
       end
   end

   if ~cartProd && ~(size(k,1) == size(ttm,1))
       error(['PricingModelClass:GetPrices: When not using cartesian',...
              ' product the moneyness and time-to-expiry vectors must be', ...
              ' of the same length.']);
   end
   
   if cartProd && (any(size(unique(k)) ~= size(k)) ...
               || any(size(unique(ttm)) ~= size(ttm)))
        error(['PricingModelClass:GetPrices: The moneyness and expiry ',...
               'vectors must be unique when a cartesian product is ',...
               'requested.']);
   end

   settings = obj.pricerSettings;
   ttmUniq = sort(unique(ttm));
    
   % If we are to aggregate prices over several runs then call this function 
   % recursively (can be useful to limit memory usage):
    if isa(settings,'MonteCarloPricerSettingsClass') && settings.nRepeat > 1 ...
            && ~specialRun
        
        if use_existing_numbers
            error(['PricingModelClass:GetPrices: Having ',...
                'use_existing_numbers = true is not ',...
                'compatible with obj.pricerSettings.nRepeat > 1.']);
        end

        % Check for any old random numbers:
        if ~isempty(settings.random)
          warning(['PricingModelClass: GetPrices: When using the ', ...
                 '''nRepeat'' parameter of the ',...
                 'MonteCarloPricerSettingsClass there must not be any ', ...
                 'random numbers stored in the object before running. ',...
                 'Clearing existing random numbers stored in ',...
                 'obj.pricerSettings.random in order to simulate new ones.']);
          settings.random = [];
        end

        % Run recursively:
        nRep = settings.nRepeat;
        
        % Change or add priceType to be 'price':
        price_type_input_exists = false;
        for i=1:size(varargin,2)
            if isstring(varargin{i}) && strcmpi(varargin{i},'priceType')
                varargin{i+1} = 'price';
                price_type_input_exists = true;
            end
        end
        if ~price_type_input_exists
            varargin{length(varargin)+1} = 'priceType';
            varargin{length(varargin)+1} = 'price';
        end
        
        % Set parameters for 'special' run:
        varargin{length(varargin)+1} = 'specialRun';
        varargin{length(varargin)+1} = true;
        
        [p,iv] = deal([]);
        for i=1:nRep
            [p_new,k_mat,ttm_mat,idxCall,se_new] = obj.GetPrices(k,ttm,...
                                                        cartProd,varargin{:});
                                                    
            % Apply checks:
            if any(any(isnan(p_new))) || any(any(isnan(se_new)))
                error(['PricingModelClass:GetPrices: Estimated a NaN ',...
                       'price and/or standard error.']);
            end
            
            % Accumulate:
            if i == 1
                p = p_new;
                if standardErrors;se = se_new;end
            else
                p = p + p_new;
                if standardErrors;se = se + se_new;end
            end
            
        end
        
        p = p ./ nRep;
        
        if standardErrors
            se = (se ./ nRep)./sqrt(nRep);
        end
        
    else
    % Compute prices etc. (non-recursive call):
        
        %% Prepare random numbers:
        if isa(settings,'MonteCarloPricerSettingsClass')
           if ~use_existing_numbers
               if ~isempty(settings.random)
                  warning(['PricingModelClass:GetPrices: Clearing existing ',...
                    'random numbers in object in order to simulate new ones.']);
               end
               settings.ClearNumbers();
               obj.GenNumbersForMC(ttmUniq);
               clearNumbersAfterRun = true;
           elseif use_existing_numbers && ~isempty(settings.random)
               clearNumbersAfterRun = false;
           elseif use_existing_numbers && isempty(settings.random) 
             error(['PricingModelClass:GetPrices: ''use_existing_numbers''', ...
                   ' was set to true but there is no random numbers stored',...
                   ' in the object. Suggestions: Either (1) set ', ...
                   '''use_existing_numbers'' to false or (2) call the ',...
                   '''GenNumbersForMC'' method with the relevant expiries. ',...
                   'Then call the ''GetPrices'' method again.']);               
           end
        end

        %% Group the expiries appropriately:
        if isa(settings,'MonteCarloPricerSettingsClass')
            % We group the expiries according to how many steps we simulate:
            idxn = settings.GetNumSteps(ttm);
            uniq_grp = unique(idxn);
            
        elseif strcmpi(class(obj.pricerSettings),'FourierPricerSettingsClass')
            % Each expiry is computed separately:
             [~,~,idxn]= unique(ttm);
             uniq_grp = unique(idxn);
        end

        %% Compute prices 
        % ... and do so separately for each expiry group:
        [p,iv] = deal([]);
        for i=1:size(uniq_grp,1)
            % Select appropriate expiries and moneyness values:
            idxKeep = idxn == uniq_grp(i);
            ttm_sub = ttm(idxKeep);
            
            if isa(obj.pricerSettings,'FourierPricerSettingsClass')
                ttm_sub = unique(ttm_sub);
                cartProd_sub = true;
            else
                cartProd_sub = cartProd;
            end

            if ~cartProd;k_sub = k(idxKeep);else;k_sub = k;end

            % Run pricing algorithm for subset of contracts:
            [p_tmp,idxCall_tmp,se_tmp,iv_tmp] = obj.GetPricesSub(...
                                                k_sub,ttm_sub,...
                                                cartProd_sub,...
                                                standardErrors);

            % Initialize (required) result matrices:
            if i == 1
                % Find matrix size:
                if ~cartProd
                    output_size = size(ttm);
                else
                    output_size = [size(k,1),size(ttm,1)];
                end
                
                % Initialize:
                if ~isempty(p_tmp);p = NaN(output_size);end
                if ~isempty(idxCall_tmp);idxCall = NaN(output_size);end
                if ~isempty(se_tmp);se = NaN(output_size);end
                if ~isempty(iv_tmp);iv = NaN(output_size);end
                [k_mat,ttm_mat] = deal(NaN(output_size));
            end

            % Store results:
            if cartProd
                k_mat(:,idxKeep') = repmat(k_sub,1,sum(idxKeep));
                ttm_mat(:,idxKeep') = repmat(ttm_sub',size(k_sub,1),1);
                if ~isempty(p);p(:,idxKeep')=p_tmp;end
                if ~isempty(idxCall);idxCall(:,idxKeep')=idxCall_tmp;end
                if ~isempty(se);se(:,idxKeep')=se_tmp;end
                if ~isempty(iv);iv(:,idxKeep')=iv_tmp;end
            else
                k_mat(idxKeep) = k_sub;
                ttm_mat(idxKeep) = ttm_sub;
                if ~isempty(p);p(idxKeep)=p_tmp;end
                if ~isempty(idxCall);idxCall(idxKeep)=idxCall_tmp;end
                if ~isempty(se);se(idxKeep)=se_tmp;end
                if ~isempty(iv);iv(idxKeep)=iv_tmp;end
            end
        end

        if isa(obj.pricerSettings,'MonteCarloPricerSettingsClass')...
            && clearNumbersAfterRun
            settings.ClearNumbers();
        end

        if specialRun
            pricesOrIV = p;
            return;
        end
    end

    %% Transform output as requested:
    y = obj.y.Eval(ttm_mat);
    q = obj.q.Eval(ttm_mat);
    F = obj.s0.*exp((y-q).*ttm_mat);
    K = F.*exp(k_mat);
    
    % Create implied volatility surface:
    if (strcmpi(priceType,'implied_volatility') || ...
       strcmpi(priceType,'implied_volatility_surface')) ...
       && isempty(iv)
        % Compute implied volatilities
        iv = blsimpv_with_negative_rates(obj.s0,K,y,ttm_mat,q,p,idxCall);  
    end

    switch priceType
        case 'implied_volatility'
            pricesOrIV = iv;
            idxCall = [];
        case 'implied_volatility_surface'
            if cartProd
                pricesOrIV = ImpliedVolatilitySurfaceClass(obj.s0,k_mat,...
                                                           ttm_mat,iv,...
                                                           obj.y,obj.q);
            else
                pricesOrIV = ImpliedVolatilitySurfaceClass(obj.s0,k_mat,...
                                                           ttm_mat,iv,...
                                                           obj.y,obj.q);
            end
            idxCall = [];
        case 'price'
            % Convert prices to the appropriate option type: 
            if ~islogical(optionType)
                switch optionType
                    case 'otm'
                        convToCall = (k_mat>=0);
                    case 'call'
                        convToCall = true(size(k_mat));
                    case 'put'
                        convToCall = false(size(k_mat));
                end
            else
                convToCall = optionType;
            end

            if ~isempty(p)
                % Is call but should be put:
                convPut = ~convToCall & idxCall;
                
                % Is put but should be call:
                convCall = convToCall & ~idxCall;

                % Convert:
                p(convPut) = p(convPut) ...
                  - obj.s0.*exp(-q(convPut).*ttm_mat(convPut)) ...
                  + K(convPut).*exp(-y(convPut).*ttm_mat(convPut));
                p(convCall) = p(convCall) ...
                  + obj.s0.*exp(-q(convCall).*ttm_mat(convCall)) ...
                  - K(convCall).*exp(-y(convCall).*ttm_mat(convCall));

            elseif ~isempty(iv)
                % Use the Black-Scholes formula to get prices:
                p = NaN(size(iv));
                p(convToCall) = bscall(obj.s0,K(convToCall),y(convToCall),...
                                       ttm_mat(convToCall),iv(convToCall).^2,...
                                       q(convToCall));    
                p(convToCall) = bsput(obj.s0,K(convToCall),y(convToCall),...
                                      ttm_mat(convToCall),iv(convToCall).^2,...
                                      q(convToCall));
            else
                error(['PricingModelClass:GetPrices: Unexpected output from',...
                       ' class method ''GetPricesSub'' which should either',...
                       ' return prices or implied volatilities (or both).', ...
                       ' None was returned.']);
            end

            idxCall = convToCall;
            pricesOrIV = p;
    end
    
    % Convert standard errors from prices to implied volatilities if needed:
    if standardErrors && (strcmpi(priceType,'implied_volatility') ...
            || strcmpi(priceType,'implied_volatility_surface'))
        % Use the delta method to get standard errors in implied volatility:
        bs_vega = BSGreek('vega',[],obj.s0,K,y,ttm_mat,iv,q);
        se = se./bs_vega;
    end

end        
[ivSurf, optPrices, stdErrors] = MonteCarloEstimation(obj,k,t,cartProd,...
                                  paths,optType,condMC,cv,retStdErrors,...
                                  antithetic,prec)        
function aStar = GetOptimalFourierAlpha(obj,K,T)
% Description: Returns an optimal dampening parameter as suggested in 
% (Lord and Kahl, 2006). We assume zero interest rate and dividend and also 
% an initial asset price of 1.
%
% Remarks:
%   o We use the rule of thumb suggested in (Lord and Kahl, 2006) where we
%     for out-of-the-money calls only look for dampening parameters (alpha's)
%     above than 0 and for out-of-the-money puts only look for values below
%     -1.
%
% Parameters:
%   K: [Nx1 real] Strikes.
%   T: [1x1 real] Expiry.
%
% Output:
%   aStar: [Nx1 real] Optimal dampening parameter.
%
% References:
%   o Lord, R. and Kahl, C., Optimal Fourier Inversion in Semi-Analytical 
%     Option Pricing. 2006, Tinbergen Institute Discussion Paper No. 2006-066/2.
%
   

    %% Find valid range of alpha's to optimize over:
    aStar = NaN(size(K));
    idxCall = K>=1;
    use_default = false;
    [aMin,aMax] = deal(NaN);
    try 
        [aMin, aMax] = obj.GetMomentExplosion(T);
    catch
        use_default = true;
    end
    if isnan(aMin) || isnan(aMax) || abs(aMin)==Inf || abs(aMax)==Inf
        use_default = true;
    end
    if use_default
       warning(['PricingModelClass:GetOptimalFourierAlpha: Something went',...
                ' wrong while computing the moment bounds. Defaulting to a',...
                ' dampening parameter of 1/2 for out-of-the-money calls and',...
                ' -1/2 for out-of-the-money puts.']);
        % Probably assumptions were not met, we set a default:
        aStar(idxCall) = 0.5;
        aStar(~idxCall) = -0.5;
        return;
    end
    alphaMin = aMin - 1;
    alphaMax = aMax - 1;

    % Define functions:
    if isa(obj,'rHestonClass')
        xiVals = obj.CharacteristicFun(T,1,'only_final_timepoint',[],true);
        phi = @(u)(obj.CharacteristicFun(T,u,'only_final_timepoint',xiVals));
    else
        phi = @(u)(obj.CharacteristicFun(T,u));
    end

    if isa(obj,'rHestonClass') ...
            && strcmpi(obj.charFunSettings.method,'RationalApprox')
    % In this case we need to limit the alpha bounds further, see
    % (Gatheral and Radoicic, 2019).
        alphaMin = max([alphaMin;-1]);
        alphaMax = min([(1 - obj.rho.^2).^(-1) - 1;alphaMax]);
    end

    %% Define objective function:
    psi = @(v,alpha) ( real(phi( v - (alpha + 1)*1i  ) ...
                        ./ ( alpha.^2 + alpha - v.^2 + 1i*(2*alpha + 1).*v) ) );
    PSI = @(alpha,K)( -alpha*log(K).' + 0.5*log( psi(0,alpha).^2 ) );  

    %% Find optimal alphas:
    
    % Optimization settings:
    eps = 0.0001;
    options = optimset('Display','off','MaxIter',10^6,'MaxFunEvals',10^6);
    
    % Loop over the strikes:
    for i=1:size(K,1)
        
        % Adjust the optimization range according to the suggested
        % 'rule-of-thumb' and set the initial guess:
        if K(i) >= 1
            lb = 0 + eps;
            ub = alphaMax - eps;
            alpha0 = mean([0,alphaMax]);
        else
            lb = alphaMin + eps;
            ub = -1 - eps;
            alpha0 = mean([-1,alphaMin]);
        end
        
        % Readjust the initial guess again if the optimization function
        % cannot properly evaluate the initial guess:
        if isnan(PSI(alpha0,K(i))) || abs(PSI(alpha0,K(i))) == Inf
            % Try a different choice:
            if K(i) >= 1
                alpha0 = alphaMax ./ 10;
            else
                alpha0 = -1 + (alphaMin - (-1)) ./ 10;
            end

            % If there is still a problem we use the default value as an 
            % initial guess:
            if isnan(PSI(alpha0,K(i))) || abs(PSI(alpha0,K(i))) == Inf
                alpha0 = (K(i) >= 1)*0.5 - (K(i) < 1)*0.5;
                if alpha0 < 0
                    lb = -1 + eps;
                    ub = 0 - eps;
                end
            end
        end
        
        % Solve the optimization problem:
        large_number = 10^(20);
        [aOpt,~,flag] = fminsearch(@(x)(PSI(x,K(i))*(1 - (x < lb | x > ub))...
                              + (x < lb | x > ub)*large_number),alpha0,options);

        if flag ~= 1
            aStar(i) = (K(i) >= 1)*0.5 - (K(i) < 1)*0.5;
            warning(['PricingModelClass:GetOptimalFourierAlpha: Something ',...
                     'went wrong when searching for an optimal alpha ',...
                     'for the strike ', num2str(K(i)), '. Defaulting to a ',...
                     'dampening parameter of ', num2str(aStar(i))]);
        else
            aStar(i) = aOpt;
        end
        
    end


end
function [lBound, uBound] = GetMomentExplosion(obj,T,maxIter)
% Description: Returns the lower and upper moment explosions at a given
% future time point T. That is, the function will return a value 'lBound' s.t. 
% E[S(T)^(mu)] is infinite for all mu <= lBound and also return a value
% 'uBound' s.t. E[S(T)^(mu)] is infinite for all mu >= uBound. Here S(t)
% denotes the asset price.
%
% Remark: The pricing model must implement the following methods:
%   o MomentExplosionInitialGuess: A function with no arguments that should
%     return a valid initial guesses for the moment explosions.
%   o GetExplosionTime: Should take a moment as input and then return the
%     moment explosion time. This method should be vectorized.
%
% Parameters:
%    T:         [Nx1 real] Future time points to consider the distribution of the
%               asset price at.
%   maxIter:    [1x1 integer (optional)] Maximum number of iterations to take,
%               default is 10^5.
% 
% Output: 
%   lBound: [Nx1 real] Lower bounds for moment explosions.
%   uBound: [Nx1 real] Upper bounds for moment explosions.
%

   if ~any(strcmpi(methods(obj),'MomentExplosionInitialGuess'))
       error(['PricingModelClass:GetMomentExplosion: Could not find method',...
              ' ''MomentExplosionInitialGuess''.']);
   end
   if ~any(strcmpi(methods(obj),'GetExplosionTime'))
       error(['PricingModelClass:GetMomentExplosion: Could not find method',...
              ' ''GetExplosionTime''.']);
   end   

   % Set the error tolerance:
   eps = 0.0001;
   
   % Maximum number of steps:
   if ~exist('maxSteps','var') || isempty(maxIter)
       maxIter = 10^5;
   end
   
   [lBound,uBound] = deal(NaN(size(T)));
   for i=1:size(T,1)
       % Get initial guesses
       [omegaLow, omegaHigh] = obj.MomentExplosionInitialGuess();
        omegaLow = omegaLow + eps;
        omegaHigh = omegaHigh - eps;
       
       % Set the initial step size:
       dT = 1;
       
       % Reset the number of steps taken:
       nIter = 0;

       % Lower bound:
       errFun = @(omega)(T(i)-obj.GetExplosionTime(omega));
       sgn = sign(errFun(omegaLow));
       errLatest = errFun(omegaLow);
       while abs(errLatest) > eps
           % Take new step:
           omegaLow = omegaLow + sgn*dT;
           nIter = nIter + 1;
           if nIter > maxIter
               error(['PricingModelClass:GetMomentExplosion: ',...
                      'Maximum number of iterations reached.']);
           end
           while sign(errFun(omegaLow)) ~= sign(errLatest)
               % We went too far, so step back:
               omegaLow = omegaLow - sgn*dT;
               % Adjust dT
               dT = dT/10;
               % Try smaller step:
               omegaLow = omegaLow + sgn*dT;
               % Check if within error tolerance:
               if abs(errFun(omegaLow)) < eps
                   break;
               end
           end
           errLatest = errFun(omegaLow);
       end
       lBound(i) = omegaLow;

       % Reset step size:
       dT = 1;
       
       % Reset the number of steps taken:
       nIter = 0;       

       % Upper bound:
       errFun = @(omega)(T(i)-obj.GetExplosionTime(omega));
       sgn = -sign(errFun(omegaHigh));
       errLatest = errFun(omegaHigh);
       while abs(errLatest) > eps
           % Take new step:
           omegaHigh = omegaHigh + sgn*dT;
           nIter = nIter + 1;
           if nIter > maxIter
               error(['PricingModelClass:GetMomentExplosion: ',...
                      'Maximum number of iterations reached.']);
           end           
           while sign(errFun(omegaHigh)) ~= sign(errLatest)
               % We went too far, so step back:
               omegaHigh = omegaHigh - sgn*dT;
               % Adjust dT
               dT = dT/10;
               % Try smaller step:
               omegaHigh = omegaHigh + sgn*dT;
               % Check if within error tolerance:
               if abs(errFun(omegaHigh)) < eps
                   break;
               end
           end
           errLatest = errFun(omegaHigh);
       end    
       uBound(i) = omegaHigh;
   end

end
end
end

