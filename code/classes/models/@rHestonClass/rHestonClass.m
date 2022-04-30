classdef rHestonClass < PricingModelClass
% Description: Implements the rough Heston model of (El Euch et al., 2019).
%
% We briefly explain the model: Let r(t), respectively delta(t), denote 
% the deterministic risk-free interest rate and the continuous proportional 
% dividend yield. We then assume that the asset price S(t) has dynamics
%
% dS(t) = S(t)(r(t) - delta(t))dt + S(t)sqrt(V(t))dW_2(t)
%
% under the risk-neutral measure where W_2(t) is Brownian motion and V(t) is
% the instantaneous variance. We assume that
%
%   V(t) = xi0(t) + int_0^t K(t-s)*nu*sqrt(V(s)) dW_1(s)
%
% where xi0 is a deterministic function, W_1 another Brownian motion such that
% dW_1(t)dW_2(t) = rho dt with rho between -1 and 1, nu > 0, and
%
%   K(t) = (1/gamma(H+1/2))*t^(H-1/2),
%
% where 0 < H < 1/2. For the implementation, we moreover assume that xi0 has 
% the form
%
%   xi0(t) = v0 + int_0^t K(t-s) theta(s) ds
%
% where v0 >= 0 and theta is a piecewise constant deterministic function
% which must be chosen such that
%
% 	theta(t)dt + (v0/gamma(1/2-H))t^{-H-1/2}dt
%
% is a non-negative measure.
%
% Properties:
%   o H:                [1x1 real] Hurst exponent. 
%   o rho:              [1x1 real] Correlation parameter.
%   o eta:              [1x1 real] Volatility-of-volatility parameter.
%   o v0:               [1x1 real] Initial instantaneous variance.
%   o theta:            [1x1 CurveClass] Curve for setting the term-structure.
%                       We require obj.theta.interpolation = 'flat' and 
%                       obj.theta.extrapolation = 'flat'.
%   o charFunSettings:  [1x1 struct] Settings for computing the characteristic 
%                       function. Read the description for the class method
%                       'CharacteristicFun' to see how this object should
%                       be specified.
%
% More properties are inherited from the PricingModelClass. The most important 
% being the obj.pricerSettings property where the settings for the pricing 
% algorithm are set. You should consult the PricingModelClass for an 
% explanation of how that object is to be interpreted. Below we only very 
% briefly explain what settings are possible within the rHeston model:
%
% The obj.pricerSettings must be of the FourierPricerSettingsClass type
% as only pricing with fourier methods is implemented. Other than that there are
% no restrictions on what settings can be chosen among those explained in
% the description for the FourierPricerSettingsClass. The following additional 
% restrictions then apply:
%   o The obj.pricerSettings.alpha property must be set to a (fixed) 
%     scalar value.
%   o The obj.pricerSettings.transform_domain property must be set to false.
%   o The obj.upper_bound_integration property must be set to some value.
%
% References:
%   o El Euch, O., Gatheral, J. and Rosenbaum, M., Roughening Heston. Risk, 
%	  May 2019, pp. 84-89.
%
    
properties
    H
    rho
    nu
    v0
    theta    
    charFunSettings
end

methods
   function obj = rHestonClass(varargin)
    % Description: Constructor. 
    % 
    % Parameters: Inputs must be given in name-value pairs corresponding to the 
    % object properties. Note that some of these properties are inherited from 
    % the PricingModelClass.
    %
    % Note also that default values may be set if some properties are not
    % specified. Also, if obj.theta is set to a scalar it is automatically 
    % converted to a CurveClass object as a flat curve equalling that scalar.
    %
    % Output:
    %   [1x1 rHestonClass] The object.
    %
    % Examples: 
    %   o model = rHestonClass('v0',0.2^2,'H',0.1,'rho',-0.6,'nu',0.2,'theta',0);
    % 
    
        obj.ParseConstructorInputs(varargin{:});

        % Set default price estimation settings:
        if isempty(obj.pricerSettings)
            intFun = @(f,a,b)(integral(f,a,b,'ArrayValued',true));
            obj.pricerSettings = FourierPricerSettingsClass('alpha',-0.5,...
                   'transform_domain',false,'upper_bound_integration',Inf,...
                   'integration_function',intFun,...
                   'integration_function_allows_vector_valued',true,...
                   'throw_error_on_negative_time_value',false,...
                   'throw_error_on_integration_warning',false);
        end

        % Set default settings for computing the characteristic function:
        if isempty(obj.charFunSettings)
            cfset = struct;
            cfset.method = 'RationalApprox';
            cfset.n = 200;
            obj.charFunSettings  = cfset;
        end

        % Set or adjust the theta curve if needed:
        if isprop(obj,'theta') && isempty(obj.theta)
            obj.theta = CurveClass('gridpoints',1,'values',0);
        end
        if isprop(obj,'theta') && isnumeric(obj.theta)
            obj.theta = CurveClass('gridpoints',1,'values',obj.theta);
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
   %    k:        [Nx1 real] Log-moneyness.
   %    T:        [Mx1 real] Expirations. Remark: Function currently only 
   %              supports M = 1.
   %    cartProd: [1x1 logical] If true then we return prices for 
   %              cartesian product of k and ttm vectors. Else we 
   %              assume N = M and return prices for each element. Remark:
   %              Function currently only supports cartProd = true.
   %    retSE:    [1x1 logical] Allows for standard errors to be returned.
   %              Functionality is currently not supported. Therefore it must
   %              be set to false.
   %
   % Output: 
   %    p:       [Nx1 or NxM real] Prices of either call or put options.
   %    idxCall: [Nx1 or NxM logical] Entry is true if price is of a call 
   %             option, otherwise it is for a put.
   %    se:      [empty] Standard errors.
   %    iv:      [empty] Black-Scholes implied volatilities.
   %

       if ~CheckNonNegReqTheta(obj.v0,obj.H,obj.theta.gridpoints,obj.theta.values)
           error(['rHestonClass:GetXi0: ''obj.theta'' does not satisfy the ',...
                  'non-negativity requirement']);
       end   
   
       if retSE
          error('rHestonClass:GetPricesSub: Standard errors are not supported.');
       end

       if size(T,1) > 1 || ~cartProd
          error(['rHestonClass:GetPricesSub: Function only supports expiry ',...
                  'inputs of size [1x1] and cartesian product must also be ',...
                  'set to true.']);
       end

       [stdErrors, iv] = deal([]);

        p = FourierPricing(1,k,T,obj);
        idxCall = true(size(k,1),1);

   end
   function xiVals = GetXi0(obj,t)
   % Description: Computes xi0(t).
   %
   % Parameters:
   %    t:  [Nx1 real] Time points. Must be sorted in ascending order.
   %
   % Output:
   %    xiVals: [Nx1 real] xi0(t).
   %
       
        % Validate setup:
        if ~issorted(t)
            error('rHestonClass:GetXi0: ''t'' must be sorted in ascending order.');
        end

        if ~(strcmpi(obj.theta.interpolation, 'flat') ...
                && strcmpi(obj.theta.extrapolation, 'flat'))
            error('rHestonClass:GetXi0: ''obj.theta'' must be piecewise constant.');
        end        
		
		if size(t,2) > 1
			error('rHestonClass:GetXi0: Input must be of size Nx1.');
		end
        
        if ~CheckNonNegReqTheta(obj.v0,obj.H,obj.theta.gridpoints,obj.theta.values)
            error(['rHestonClass:GetXi0: ''obj.theta'' does not satisfy the ',...
                   'non-negativity requirement']);
        end
        
        % Compute values:
        xiVals = zeros(size(t));
        for i=1:size(t,1)
            idxGrid = obj.theta.gridpoints <= t(i);
            lb = [0; obj.theta.gridpoints(idxGrid)];
            ub = [obj.theta.gridpoints(idxGrid); t(i)];
            theta_val = obj.theta.Eval(0.5*(lb + ub));
            K_integrals = (1/gamma(obj.H+3/2))*((t(i)-lb).^(obj.H+1/2)...
                                               -(t(i)-ub).^(obj.H+1/2));
            xiVals(i) = obj.v0 + sum(theta_val.*K_integrals);
        end
        
   end
   
   function [val, ts] = CharacteristicFun(obj,T,u,outputType,xiVals,retXiOnly)
   % Description: Implements the characteristic function of the log spot price 
   % here assuming zero interest rates and dividends and an initial asset price 
   % of 1.
   %
   % To be specific the function returns
   %
   %    E[e^{i*u*log(S(T))}]
   %
   % where dS(t) = sqrt(V(t))S(t)dW_2(t), S(0) = 1, is the (risk-neutral) asset 
   % dymamics under the rough Heston model with the already mentioned
   % simplifying assumptions.
   %
   % To specify which computational method is used we set the
   % obj.charFunSettings.method object property. Currently the only option is
   % to set it to 'RationalApprox' in which case we use the method from
   % (Gatheral and Radoicic, 2019). For this method the object property 
   % obj.charFunSettings.n specifies how many steps (in total) should be used to 
   % approximate the time integral in equation (1.2) of that paper. If
   % multiple time points are requested (M > 1 below) this is the number of 
   % steps used until the largest time point.
   %
   % Parameters:
   %    T:          [Mx1 real] Future time point(s).
   %    u:          [Nx1 real or complex] Argument to characteristic function.
   %    outputType: [1x1 string (optional)] This parameter is only in effect if
   %                 M = 1, i.e. T is a scalar. In that case the options are as 
   %                 below.
   %                    o 'only_final_timepoint': Default value. See the output
   %                      description for an explanation of the behaviour.
   %                    o 'incl_intermediate_timepoints': See the output
   %                      description for an explanation of the behaviour.
   %    xiVals:     [1xL (optional)] (Added for performance reasons) The vector 
   %                of forward variances needed for the computations. Must be
   %                obtained by first calling the function with retXiOnly =
   %                true. If not specified the forward variances are
   %                automatically computed.
   %    retXiOnly:  [1x1 logical (optional)] (Added for performance reasons) If 
   %                set to true (default is false) the output 'val' will contain
   %                the vector of forward variances needed for the computation. 
   %                This can then be used  for repeated calls to the function 
   %                using the parameter  'xiVals' thus avoiding repeated 
   %                evaluations of the forward variance curve.
   %                    
   % Output: 
   %    val: [NxM or NxL, real or complex] If M > 1 then we here return the
   %         value of the characteristic function for each combination of the
   %         the entries in the u and T vectors. The same is true if M = 1
   %         and outputType = 'only_final_timepoint'. If M = 1 and 
   %         outputType = 'incl_intermediate_timepoints' then we instead
   %         return the characteristic function for each u and then each
   %         time point between 0 and T with a step length of delta = 
   %         T / obj.charFunSettings.n property. If there are L such timepoints 
   %         the output will therefore be of size NxL.
   %         Remark: If M > 1 we compute time step lengths as delta =
   %         max(T) / obj.charFunSettings.n. Any entries in T that do not
   %         exactly fit into a time grid with this step length will then
   %         be truncated down to the nearest grid point. If M = 1 and
   %         outputType = 'incl_intermediate_timepoints' it is then also
   %         exactly the time points delta, 2*delta, ..., max(T) for which
   %         we return the values.
   %         Remark: If retXiOnly = true the behaviour will be different,
   %         see the description of that parameter.
   %    ts:  [empty or 1xL real] Empty unless outputType =
   %         'incl_intermediate_timepoints' in which case it will contain
   %         the timepoints corresponding to the output 'val'.
   %
   % References:
   %   o Gatheral, J. and Radoicic, R., Rational Approximation of the rough 
   %     Heston solution. 2019, International Journal of Theoretical and Applied 
   %     Finance, 22(3), 1950010.
   %
   
        if ~CheckNonNegReqTheta(obj.v0,obj.H,obj.theta.gridpoints,obj.theta.values)
            error(['rHestonClass:GetXi0: ''obj.theta'' does not satisfy the ',...
                   'non-negativity requirement']);
        end   
   
        ts = [];
       
        % Compute step size:
        Tmax = max(T);
        n = obj.charFunSettings.n;
        delta = Tmax/n; 
        t = (0:delta:Tmax)';

        if exist('retXiOnly','var') && ~isempty(retXiOnly) && retXiOnly
            val = obj.GetXi0(t(1:end-1));
            return;
        end

       if ~exist('xiVals','var') || isempty(xiVals)
           xiVals = obj.GetXi0(t(1:end-1));
       end
       
       if ~exist('outputType','var') || isempty(outputType)
           outputType = 'incl_intermediate_timepoints';
       end

       % Solve fractional riccati equation:
       method = obj.charFunSettings.method;
       if strcmpi(method,'RationalApprox')
          g = obj.SolveRiccatiRationalApprox(u,t(1:end-1),obj.H,obj.nu,obj.rho);
       else
          error(['rHestonClass:CharacteristicFun: Method ''', method, ...
                  ''' is not supported.']);
       end

       % Equation (1.2) in (Gatheral and Radoicic, 2019):
       if size(T,1) > 1 || strcmpi(outputType,'incl_intermediate_timepoints')
           val = exp(delta.*cumsum(bsxfun(@times,flipud(xiVals.'),g),2));
       else
           val = exp(delta.*sum(bsxfun(@times,flipud(xiVals.'),g),2));
       end

       % Here we modify the output depending on what the user requested:
       if size(T,1) > 1
           % Truncate to fit into the grid of delta's:
           TAdj = floor(n*T) ./ n; 
           idxKeep = ismembertol(t(2:end),TAdj);
           val = val(:,idxKeep);
       elseif strcmpi(outputType,'only_final_timepoint')
           val = val(:,end);
       elseif strcmpi(outputType,'incl_intermediate_timepoints')
           ts = t(2:end);
       else
         error(['rHestonClass:CharacteristicFun: Output type ''',outputType, ...
                 ''' is not supported.']);
       end

   end
   Dalphah = SolveRiccatiRationalApprox(~,a,t,H,nu,rho);
end
end

