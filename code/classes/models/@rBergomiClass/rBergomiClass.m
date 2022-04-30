classdef rBergomiClass < PricingModelClass & handle
% Description: Implements the rough Bergomi model of (Bayer et al., 2016).
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
% V(t) = xi(t)*exp(eta*sqrt(2H)*int_0^t(t-s)^{H-1/2}dW_1(s)-0.5*eta^2*t^{2H})
%
% with W_1 another Brownian motion s.t. dW_1(t)dW_2(t) = rho dt for a 
% -1 <= rho <= 1. Also, xi(t) is a deterministic function, 0 < H < 1/2 and 
% eta > 0.
%
% Properties:
%   o H:    [1x1 real] Hurst exponent.
%   o rho:  [1x1 real] Correlation parameter.
%   o eta:  [1x1 real] Volatility-of-volatility parameter.
%   o xi:   [1x1 CurveClass] Forward variance curve.
%
% More properties are inherited from the PricingModelClass. The most important 
% being the obj.pricerSettings property where the settings for the pricing 
% algorithm are set. Since only Monte Carlo pricing is implemented for this
% class this property must be of the MonteCarloPricerSettingsClass type. You 
% should consult the description of that class for more details on the possible 
% settings. There are no further restrictions on what pricing settings are 
% allowed than those explained in that class.
%
% References:
%   o Bayer, C., Friz, P. and Gatheral, J., Pricing under rough volatility.
%     Quantitative Finance, 2016, 16(6), 887-904.
%

properties
   H               
   rho             
   eta            
   xi              
end
    
methods
   function obj = rBergomiClass(varargin)
    % Description: Constructor. 
    % 
    % Parameters: Inputs must be given in name-value pairs corresponding to 
    % the object properties. Note that some of these properties are inherited 
    % from the PricingModelClass.
    %
    % Note also that default values may be set if some properties are not
    % specified. Also, if the forward variance curve obj.xi is set to a
    % scalar it is automatically converted to a CurveClass object with a
    % flat value of that scalar.
    %
    % Output:
    %   [1x1 rBergomiClass] The object.
    %
    % Examples: 
    %   o rBergomiClass('H',0.1,'rho',-0.9,'eta',2,'xi',0.2^2)
    %   o rBergomiClass('H',0.1,'rho',-0.9,'eta',2)
    %   o xi = CurveClass('gridpoints',[0.1;1],'values',[0.1^2;0.2^2])
    %     rBergomiClass('H',0.1,'rho',-0.9,'eta',2,'xi',xi)
    %

        % Set object properties:
        obj.ParseConstructorInputs(varargin{:});

        % Set default settings for pricing algorithm:
        if isempty(obj.pricerSettings)
            % Price estimation settings:
            priceEst = struct;
            priceEst.control_variate = 'asset_price';
            priceEst.antithetic = true;
            priceEst.conditional_monte_carlo = true;
            priceEst.option_type = 'otm';

            % Simulation settings:
            sim = struct;
            sim.scheme = 'hybrid';
            sim.conv_method = 'optimal';
            n = [50000;25000;12500;6400;3200;500];
            tn = [0.004;0.008;0.016;0.032;0.2;Inf];                

            obj.pricerSettings = MonteCarloPricerSettingsClass(...
                                     'n',n,'tn',tn,'N',100000,...
                                     'price_estimation',priceEst,...
                                     'simulation',sim);
        else
            if isempty(obj.pricerSettings.tn) && size(obj.pricerSettings.n,1)==1
                obj.pricerSettings.tn = Inf;
            end
        end

        % Set or adjust the forward variance curve if needed:
        if isprop(obj,'xi') && isempty(obj.xi)
            obj.xi = CurveClass('gridpoints',1,'values',0.1);
        end
        if isprop(obj,'xi') && isnumeric(obj.xi)
            obj.xi = CurveClass('gridpoints',1,'values',obj.xi);
        end
        
        % Adjust the yield and dividend yield curve if needed:
        if isprop(obj,'y') && isnumeric(obj.y)
            obj.y = CurveClass('gridpoints',1,'values',obj.y);
        end        
        if isprop(obj,'q') && isnumeric(obj.q)
            obj.q = CurveClass('gridpoints',1,'values',obj.q);
        end

   end  
   function [p, idxCall, se, iv] = GetPricesSub(obj,k,T,cartProd,retSE)
   % Description: Computes prices of put and call options. The function 
   % is not intended to be used by an end-user but only as an auxiliary 
   % function to be used by the 'GetPrices' method of the PricingModelClass.
   %
   % Remark: 
   %    o We assume that all input expirations use the same number of
   %      steps per year (according to the obj.pricerSettings property).
   %    o Currently the outputted implied volatilities are empty. For the
   %      end-user these will be computed in the 'GetPrices' method of the
   %      PricingModelClass if requested.
   %
   % Parameters:
   %    k:        [nx1 real] Log-moneyness.
   %    T:        [mx1 real] Expirations.
   %    cartProd: [1x1 logical] If true then we return prices for the cartesian 
   %              product of k and ttm vectors. Else we assume n = m and return 
   %              prices for each element.
   %    retSE:    [1x1 logical (optional)] If set to true we return standard 
   %              errors, otherwise we do not. Default is false.
   %
   % Output: 
   %    p:       [nx1 or nxm real] Prices of either call or put options.
   %    idxCall: [nx1 or nxm logical] An element is true if the price is 
   %             of a call option, otherwise it is of a put.
   %    se:      [nx1 or nxm real] Standard errors.
   %    iv:      [empty] Black-Scholes implied volatilities. Currently left 
   %             empty.
   %
        
       iv = [];
   
       % Extract pricer settings:
       settings = obj.pricerSettings;
       optType = settings.price_estimation.option_type;
       cv = settings.price_estimation.control_variate;
       condMC = settings.price_estimation.conditional_monte_carlo;
       antithetic = settings.price_estimation.antithetic;
       scheme = settings.simulation.scheme;
       prec = settings.precision;

       if ~exist('retSE','var')||isempty(retSE);retSE=false;end

       % Validate that maturities all require the same number of steps per year:
       [~, nUse] = settings.GetNumSteps(T);
       if size(unique(nUse),1) > 1
           error(['rBergomiClass:GetPricesSub: All expiries must ', ...
                  'use the same number of steps per year.']);
       end

       % Set extra parameters/settings for the simulation part:
       if ~isa(settings.random,'RandomNumbersClass')
           extraInput = {'rndNumbers',settings.random{...
                         find(settings.randomMaxT <= max(T),1,'last')}};
       else
           extraInput = {'rndNumbers',settings.random};
       end
       extraInput = {extraInput{:},'antithetic',antithetic,'scheme',scheme};
       
       sim_variables = {};
       if ~condMC
           sim_variables = {sim_variables{:},'S'};
       end
       if condMC || strcmpi(cv,'timer_option')
           sim_variables = {sim_variables{:},'S1','QV'};
       end
       
       % Simulate paths:
       paths = obj.Simulate(settings.N, nUse(1), sort(unique(T)), ...
                            'outVars',sim_variables, extraInput{:});

       % Compute prices:
       [p, idxCall, se] = obj.MonteCarloEstimation(k,T,cartProd,paths,...
                                            optType,condMC,cv,retSE,...
                                            antithetic,prec);

   end
   function varargout = GenNumbersForSim(~,N,M,outVars,anti,onlyRetSpecs)
   % Description: Generates underlying random numbers to be used for a call 
   % to the class method 'Simulation'.
   %
   % Parameters: 
   %    N:            [1x1 integer] Number of independent paths.
   %
   %    M:            [1x1 integer] Total number of steps.
   %
   %    outVars:      [1xL cell (optional)] See the Simulate method for a
   %                  description. Short version: A list of the variables/
   %                  processes that we want the 'Simulate' method to return. 
   %                  Can be left empty, default is then {'S','sig'}.
   %
   %    anti:         [1x1 logical (optional)] Set to true if antithetic  
   %                  paths also needs to be simulated. Default is false.
   %
   %    onlyRetSpecs: [1x1 logical (optional)] If true then we just return 
   %                  the variable names and the sizes of each set of 
   %                  random numbers. See also the description under 
   %                  'Output'. The default value is false.
   %
   % Output: 
   %    If onlyRetSpecs = false the output is an object of class 
   %    'RandomNumbersClass' where the 'numbers' property is a struct with the 
   %    following members:
   %
   %        o V_Gaussians: [2xN*M or 2x(N*M/2) real] Independent standard 
   %          normal distributed numbers needed for simulation of the 
   %          instantaneous variance process.
   %
   %        o Wperp_Gaussians: [NxM or (N/2)xM real] Independent standard  
   %          normal distributed numbers needed for simulation of the  
   %          Brownian motion driving the underlying asset.
   %
   %    Remark: The sizes depend on the value of the 'anti' parameter.
   %
   %    If onlyRetSpecs = true there will be two ouputs:
   %
   %        (1) [1xL cell] The names of the members/variables that would be 
   %            in the 'RandomNumbersClass' if we had onlyRetSpecs = false. 
   %
   %        (2) [1xL cell] Each element consists of a [1x2 integer] vector
   %            giving the dimension of each variable from (1) if we had
   %            onlyRetSpecs = false.
   %
   %    See also the example below.
   %
   % Example:
   %    o model = rBergomiClass('H',0.1,'eta',2,'rho',-0.9,'xi',0.2^2);
   %      numbers = model.GenNumbersForSim(1000,100,{'S','sig'},true,false);
   %      [vars,sizes] = model.GenNumbersForSim(1000,100,{'S','sig'},true,true);
   %

       if ~exist('outVars','var') || isempty(outVars)
           outVars = {'S','sig'};
       end
       if ~exist('anti','var') || isempty(anti)
           anti = false;
       end
       if ~exist('onlyRetSpecs','var') || isempty(onlyRetSpecs)
           onlyRetSpecs = false;
       end

       if anti;Nact=N/2;else;Nact=N;end

       if onlyRetSpecs
          if any(strcmpi(outVars,'S'))
              vars = {'V_Gaussians','Wperp_Gaussians'};  
              sizes = {[2,Nact*M],[Nact,M]};
          else
              vars = {'V_Gaussians'};
              sizes = {[2,Nact*M]};
          end
          varargout = {vars,sizes};
          return;
       end

       % Generate numbers
       num = struct;
       num.V_Gaussians = normrnd(0,1,2,Nact*M);

       if any(strcmpi(outVars,'S'))
           num.Wperp_Gaussians = normrnd(0,1,Nact,M);
       end

       % Create object:
       varargout = {RandomNumbersClass('numbers',num)};

   end
   function GenNumbersForMC(obj,ttm,type)
   % Description: Generates random numbers for one or several runs of the Monte 
   % Carlo pricer and stores them in the obj.pricerSettings.random property.
   %
   % Parameters:
   %    ttm:  [Nx1 real] Expiries we wish to price using Monte Carlo.
   %    type: [1x1 string (optional)] The options are stated below:
   %    
   %            o 'Gaussians': This is the recommended choice (and the default).
   %                           Here the underlying i.i.d. standard normal random 
   %                           variables needed for running the Monte Carlo 
   %                           pricing algorithm with the expiries given in 
   %                           'ttm' are generated and stored.
   %
   %            o 'Y_and_dW1': This choice has a more narrow purpose. Here we 
   %                           pre-simulate the paths of the Y(t) process and 
   %                           the increments of the W_1(t) Brownian motion and 
   %                           store those numbers for the Monte Carlo pricing 
   %                           algorithm to use. If the pricing algorithm also 
   %                           needs any processes that are measurable with 
   %                           respect to the path of the W_perp(t) Brownian 
   %                           motion this will have to be simulated on the run 
   %                           as needed. See the class method 'HybridScheme'  
   %                           for an explanation of the notation, in
   %                           particular how the Y(t) process is defined.
   %                    

       settings = obj.pricerSettings;
       if ~isa(settings,'MonteCarloPricerSettingsClass')
           error(['rBergomiClass:GenNumbersForMC: ', ...
                  'Pricer settings must be an object of type', ...
                  ' ''MonteCarloPricerSettingsClass''.']);
       end

       if ~exist('type','var') || isempty(type)
           type = 'Gaussians';
       end

       % Generate numbers:
       if strcmpi(type,'Gaussians')
            % Find the total number of steps needed:
            [~,~,nTotalSteps] = settings.GetNumSteps(ttm);  
            
            % Find the processes required for price estimation:
            condMC = settings.price_estimation.conditional_monte_carlo;
            cv = settings.price_estimation.control_variate;
            sim_variables = {};
            if ~condMC
            	sim_variables = {sim_variables{:},'S'};
            end
            if condMC || strcmpi(cv,'timer_option')
                sim_variables = {sim_variables{:},'S1','QV'};
            end                      
            
            % Generate random numbers:
            settings.random = obj.GenNumbersForSim(settings.N,...
                                          max(nTotalSteps),sim_variables,...
                                          settings.price_estimation.antithetic);
                        
       elseif strcmpi(type,'Y_and_dW1')
           % Find the total number of steps needed but do so for each 
           % set of expiries having the same number of steps per year:
           [dtGrp,nPerYear,~] = settings.GetNumSteps(ttm);

           uniqDtGrp = unique(dtGrp);
           if settings.price_estimation.antithetic
               Nuse = settings.N/2;
           else
               Nuse = settings.N;
           end
           
           cellArr = cell(size(uniqDtGrp,1),1);
           maxTByObj = NaN(size(uniqDtGrp,1),1);
           for i=1:size(uniqDtGrp,1)
                rndobj = RandomNumbersClass();
                rndobj.numbers = obj.Simulate(Nuse, ...
                    unique(nPerYear(dtGrp==uniqDtGrp(i))),...
                    max(ttm(dtGrp==uniqDtGrp(i))), ...
                    'outVars',{'dW1','Y'},...
                    'antithetic',false, ...
                    'scheme',settings.simulation.scheme,...
                    'timePoints','all_up_till');
                rndobj.numbers.Y = rndobj.numbers.Y(:,1:end-1);
                rndobj.numbers.dW1 = rndobj.numbers.dW1(:,2:end);
                rndobj.numbers.t = [];
                cellArr{i} = rndobj;
                maxTByObj(i) = max(ttm(dtGrp==uniqDtGrp(i)));
           end
           settings.randomMaxT = maxTByObj;
           settings.random = cellArr;
       else
           error('rBergomiClass:GenNumbersForMC: Invalid input.');
       end

   end
   function paths = Simulate(obj, N, n, t, varargin)
    % Description: Simulates the rough Bergomi model assuming an initial asset
    % price of 1 and zero drift.
    %
    % Warning: This function can be quite memory intensive depending on how
    % many paths and time steps are used.    
    %
    % Parameters:
    %       N: [1x1 integer] Number of paths to simulate.
    %       n: [1x1 integer] Number of time steps per year.
    %       t: [Mx1 real] Vector of time points to return variables/processes 
    %           on. If the time points do not fit into the equidistant 
    %           simulation grid with step lengths of 1/n then the value 
    %           at such time points will be approximated by the value at the
    %           grid point just below. Important: 
    %           (1) Must be sorted in ascending order.
    %           (2) The behaviour of this parameter will be different if the 
    %               optional parameter 'timePoints' is specified - see below. 
    %
    % Optional parameters (in name-value pairs):
    %   timePoints: [1x1 string] Changes the interpretation of the input 
    %               parameter 't'. The options are explained below.
    %
    %               o 'explicit': Here t may be a vector. Values will then
    %                 be returned for all time points specified in t (exactly
    %                 as described under the 't' parameter). This is the 
    %                 default if left empty.
    %
    %               o 'all_up_till': Here t must be scalar (M=1). Values will
    %                 then be returned on all the time points 
    %                       0 < 1/n < 2/n < ... < floor(t*n)/n
    %
    %               o 'all_up_till_excl_first': Here t must be scalar (M=1). 
    %                 Values will then be returned on all the time points 
    %                           1/n < 2/n < ... < floor(t*n)/n
    %
    %               o 'all_up_till_excl_last': Here t must be scalar (M=1). 
    %                 Values will then be returned on all the time points 
    %                       0 < 1/n < 2/n < ... < (floor(t*n) - 1)/n 
    %
    %   outVars:  [1xL] Cell array with names of the stochastic processes 
    %             you want outputted. Options are: 'S', 'V', 'sig', 'Y', 'QV', 
    %             'int_sig_dW1', 'S1', 'log_S1'. See the class method
    %             'HybridScheme' for an explanation of what exactly they mean.
    %           
    %             Remark: Some of these options are (mostly) available for 
    %             either performance reasons (to avoid recomputing them) 
    %             or for a very specific purpose. 
    %
    %             If left empty the default is outVars = {'S','sig'}.
    %
    %   antithetic: [1x1 logical] If true we return antithetic sample paths 
    %               as well (still N paths in total). It is then the first
    %               N/2 paths returned that will be the original paths and
    %               the bottom half (N/2) paths that will be antithetic. 
    %               The default is antithetic = false.
    %
    %   rndNumbers: [1x1 RandomNumbersClass or empty] Object containing the 
    %               random numbers that will be used for the simulation. If  
    %               left empty we simulate them as needed. Consult the 
    %               HybridScheme class method for more details on this
    %               parameter. 
    %
    %               Important: It should be stressed that (if used) one should 
    %               be careful to set this parameter correctly to avoid 
    %               simulating paths with incorrect distributional properties. 
    %               The recommended choice is therefore to leave it empty.
    %
    %   scheme: [1x1 string] Simulation scheme to use. Currently the only
    %            allowed value is 'hybrid' (also the default) in which 
    %            case we use the hybrid scheme of (Bennedsen et al., 2017)
    %            with their kappa = 1.
    %
    % Output: 
    %   paths: [1x1 struct] Struct containing a field for each element in
    %          'outVars' as well as a field named 't' with the time points of 
    %          the simulation.
    %
    % References: 
    %   o Bennedsen, M., Lunde, A. and Pakkanen, M.S., Hybrid scheme for 
    %     Brownian semistationary procesess. Finance and Stochastics, 2017, 
    %     21(4), 931-965.
    % 

       % Parse inputs:
       p = inputParser;
       addParameter(p,'timePoints','explicit');
       addParameter(p,'outVars',{'S','sig'});
       addParameter(p,'antithetic',false);
       addParameter(p,'rndNumbers',[]);
       expectedSchemes = {'hybrid'};
       addParameter(p,'scheme','hybrid',...
                   @(x) any(validatestring(x,expectedSchemes)));
       parse(p,varargin{:});
       v2struct(p.Results);
       expectedOutVars = {'S','V','sig','Y','QV','int_sig_dW1','S1',...
                          'log_S1','dW1'};
       if sum(ismember(outVars,expectedOutVars)) ~= size(outVars,2)
           error(['rBergomiClass:Simulate: Some of the chosen output ',...
                  'variables are not supported.']);
       end

       dt = 1 / n;
       grid_pts = (1/n)*(0:floor(max(t)*n + 1))';
       idxMaxT = sum(grid_pts <= t(end),1);
       
       if strcmpi(timePoints,'explicit')
           nStepsTotal = idxMaxT - 1;
       elseif strcmpi(timePoints,'all_up_till')
           t = dt*(0:idxMaxT-1)';
           nStepsTotal = idxMaxT - 1;
       elseif strcmpi(timePoints,'all_up_till_excl_first')
           t = dt*(1:idxMaxT-1)';
           nStepsTotal = idxMaxT - 1;
       elseif strcmpi(timePoints,'all_up_till_excl_last')
           t = dt*(0:idxMaxT-2)';
           nStepsTotal = idxMaxT - 2;
       end
       
       % Simulate random variables:
       if isempty(rndNumbers)
            rndNumbers = obj.GenNumbersForSim(N,nStepsTotal,outVars,antithetic);
       end
       
       % Run simulation:
       switch scheme
           case 'hybrid'
               paths = rBergomiClass.HybridScheme(obj.H,obj.rho,obj.eta,...
                            obj.xi,n,N,antithetic,t,rndNumbers,outVars,...
                            obj.pricerSettings.simulation.conv_method,...
                            obj.pricerSettings.precision);
           otherwise
               error('rBergomiClass:Simulate: Scheme is not supported.');
       end

   end
end

methods(Static)
   paths = HybridScheme(H,rho,eta,xi,n,N,anti,retPts,rndNumbers,outVars,...
                        convMethod,prec,retAllPts)
   covM = CovMatrixHybrid(n,alpha,prec)
end
    
end

