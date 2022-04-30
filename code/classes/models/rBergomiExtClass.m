classdef rBergomiExtClass < PricingModelClass & handle
% Description: This class implements the extended rough Bergomi model.
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
% V(t) = xi(t)*V_1(t)*V_2(t)
%
% with
%
% V_1(t) = exp(  zeta*sqrt(2*alpha+1)*int_0^t(t-s)^{alpha}dW_1(s)
%              - 0.5*zeta^2*t^{2*alpha+1})
% V_2(t) = exp(  lambda*sqrt(2*beta+1)*int_0^t(t-s)^{beta}dW_2(s)
%              - 0.5*lambda^2*t^{2*beta+1})
%
% where W_1 is another Brownian motion independent of W_2 and xi(t) is a 
% deterministic function. Also, alpha and beta lie in (-1/2,1/2). 
%
% Finally note that the implementation uses the following reparameterisation of 
% the parameters zeta and lambda in terms of rho and eta:
%
%   rho := lambda / sqrt(zeta^2 + lambda^2)
%   eta := sqrt(zeta^2 + lambda^2)
%
% Note that -1 <= rho <= 1 and eta > 0.
%
% Properties:
%   eta:    [1x1 real] Volatility-of-volatility parameter.
%   rho:    [1x1 real] Correlation parameter.
%   alpha:  [1x1 real] Roughness parameter of the uncorrelated factor.
%   beta:   [1x1 real] Roughness parameter of the correlated factor.
%   xi:     [1x1 CurveClass] Initial forward variance curve.
%
% More properties are inherited from the PricingModelClass. The most important 
% being the obj.pricerSettings property where the settings for the pricing 
% algorithm are set. You should consult the PricingModelClass for an 
% explanation of how that object is to be interpreted. Below we only very 
% briefly explain what settings are possible for the extended rBergomi 
% model:
%
% The obj.pricerSettings property must be of the MonteCarloPricerSettingsClass 
% type as only Monte Carlo simulation is implemented. The following additional 
% restrictions then apply:
%
%   o The obj.pricerSettings.control_variate property can only be set to
%     'asset_price' or 'none.
%   o The obj.pricerSettings.conditional_monte_carlo property must be set
%     to false.
%

properties
    eta
    rho
    alpha
    beta
    xi
end
methods
   function obj = rBergomiExtClass(varargin)
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
    %   obj:    [1x1 rBergomiExtClass] The object.
    %
    % Examples:
    %   o model = rBergomiExtClass('eta',2,'rho',-0.9,'alpha',-0.4,...
    %                              'beta',-0.4,'xi',0.2^2);
    %

        % Set object properties:
        obj.ParseConstructorInputs(varargin{:});

        % Set default settings:
        if isempty(obj.pricerSettings)
            % Price estimation settings:
            priceEst = struct;
            priceEst.control_variate = 'asset_price';
            priceEst.antithetic = true;
            priceEst.conditional_monte_carlo = false;
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
            if isempty(obj.pricerSettings.tn) && ...
                    size(obj.pricerSettings.n, 1) == 1
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
   % function to be used by the GetPrices method of the PricingModelClass.
   %
   % Remark: 
   %    o We assume that all input expirations use the same number of
   %      steps per year (according to the obj.pricerSettings property).
   %    o Currently the outputted implied volatilities are empty. For the
   %      end-user these will be computed in the GetPrices method of the
   %      PricingModelClass if requested.
   %
   % Parameters:
   %    k:        [nx1 real] Log-moneyness.
   %    T:        [mx1 real] Expirations.
   %    cartProd: [1x1 logical] If true then we return prices for the  
   %              cartesian product of k and ttm vectors. Else we assume 
   %              n = m and return prices for each element.
   %    retSE:    [1x1 logical] If set to true we return standard errors,
   %              otherwise we do not. Default is false.
   %
   % Output: 
   %    p:       [nx1 or nxm real] Prices of either call or put options.
   %    idxCall: [nx1 or nxm logical] An element is true if the price is 
   %             of a call option, otherwise it is of a put.
   %    se:      [nx1 or nxm real] Standard errors.
   %    iv:      [nx1 or nxm real] Black-Scholes implied volatilities.
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
       
       if obj.alpha <= -0.5 || obj.alpha >= 0.5 || ...
          obj.beta <= -0.5  || obj.beta  >= 0.5 || ...
          obj.rho < -1      || obj.rho   > 1    || ...
          obj.eta <= 0      
            error(['rBergomiExtClass:GetPricesSub: One or more of ',...
                   'the model parameters are invalid.']);
       end
       
       if ~exist('retSE','var');retSE = false;end
       
       % Validate that maturities all require the same number of steps per year:
       [~, nUse] = settings.GetNumSteps(T);
       if size(unique(nUse),1) > 1
           error(['rBergomiExtClass:GetPricesSub: All expiries must ', ...
                  'use the same number of steps per year.']);
       end       
       
       % Determine the number of paths to use:
       if ~isempty(settings.N_vol_grp)
           if size(settings.N_vol_grp,1) ~= size(settings.N,1)
               error(['rBergomiExtClass:GetPricesSub: When ',...
                     'obj.pricerSettings.N_vol_grp is non-empty the size ',...
                     'should match that of obj.pricerSettings.N.']);
           end
           uniqT = unique(T);
           min_vs_rate = min(sqrt(obj.xi.Integrate(zeros(size(uniqT)),uniqT)./uniqT));
           if min_vs_rate <= settings.N_vol_grp(1)
               idxN = 1;
           elseif min_vs_rate > settings.N_vol_grp(end)
               error(['rBergomiExtClass:GetPricesSub: Property ',...
                     'obj.pricerSettings.N_vol_grp does not cover the ',...
                     'forward variance curve levels.']);               
           else
               idxN = find(min_vs_rate > settings.N_vol_grp,1,'last') + 1;
           end
           Nuse = settings.N(idxN);
       else
           Nuse = settings.N;
       end
       
       % Set extra parameters/settings for the simulation part:
       if ~isa(settings.random,'RandomNumbersClass')
           extraInput = {'rndNumbers',settings.random{...
                    find(settings.randomMaxT <= max(T),1,'last')}};
       else
           extraInput = {'rndNumbers',settings.random};
       end
       extraInput = {extraInput{:},'antithetic',antithetic,'scheme',scheme};
       
       sim_variables = {'S'};
       if condMC
           error(['rBergomiExtClass:GetPricesSub: Class does not allow for ',...
                  'conditional Monte Carlo.']);
       end
       if ~(strcmpi(cv,'asset_price') || strcmpi(cv,'none'))
           error(['rBergomiExtClass:GetPricesSub: Class only allows for ',...
                  'the control variates ''asset_price'' and ''none''.']);           
       end

       % Simulate paths:
       paths = obj.Simulate(Nuse, nUse(1), sort(unique(T)), ...
                           'outVars',sim_variables, extraInput{:});

       % Compute prices:
       [p,idxCall,se] = obj.MonteCarloEstimation(k,T,cartProd,paths,optType,...
                                                 condMC,cv,retSE,antithetic,...
                                                 prec);       

   end   
   function varargout = GenNumbersForSim(~,N,M,outVars,anti,onlyRetSpecs)
   % Description: Generates underlying random numbers to be used for a call 
   % to the class method Simulation.
   %
   % Parameters: 
   %    N:            [1x1 integer] Number of independent paths.
   %
   %    M:            [1x1 integer] Total number of steps.
   %
   %    outVars:      [1xL cell (optional)] See the Simulate method for a
   %                  description. Short version: A list of the variables/
   %                  processes that we want the Simulate method to return. 
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
   %    o If onlyRetSpecs = false the output is an object of class 
   %         'RandomNumbersClass' where the 'numbers' property is a struct
   %         with the following members:
   %
   %        o V1_Gaussians: [2xN*M or 2x(N*M/2) real] i.i.d. standard normal 
   %          random numbers needed for simulation of the instantaneous 
   %          variance process.
   %
   %        o V2_Gaussians: [NxM or (N/2)xM real] i.i.d. standard normal 
   %          random numbers needed for simulation of the Brownian motion 
   %          driving the underlying asset.
   %
   %    Remark: The sizes depend on the value of the 'anti' parameter.
   %
   %    o If onlyRetSpecs = true the output is a [1x2 cell] containing:
   %
   %        o The names of the members/variables that would be in the
   %        'RandomNumbersClass' if we had onlyRetSpecs = true 
   %
   %        o The size of the matrices associated with each variable.
   %
   % Example:
   %    model = rBergomiExtClass('alpha',-0.4,'beta',-0.4,'eta',2,...
   %                               'rho',-0.9,'xi',0.2^2);
   %    numbers = model.GenNumbersForSim(1000,100,{'S','sig'},true,false);
   %    [vars,sizes] = model.GenNumbersForSim(1000,100,{'S','sig'},true,true);
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
       
       if any(strcmpi(outVars,'S') | strcmpi(outVars,'sig')) ...
               || (any(strcmpi(outVars,'V1')) && any(strcmpi(outVars,'V2')))
           vars = {'V1_Gaussians','V2_Gaussians'};
           sizes = {[2,Nact*M],[2,Nact*M]};
       elseif any(strcmpi(outVars,'V1'))
           vars = {'V1_Gaussians'};
           sizes = {[2,Nact*M]};
       elseif any(strcmpi(outVars,'V2'))
           vars = {'V2_Gaussians'};
           sizes = {[2,Nact*M]};
       elseif any(strcmpi(outVars,'Y')) || any(strcmpi(outVars,'dW1'))
           vars = {'V1_Gaussians'};
           sizes = {[2,Nact*M]};
       end
       
       if onlyRetSpecs
          varargout = {vars,sizes};
          return;
       end

       % Generate numbers
       num = struct;
       if any(strcmpi(vars,'V1_Gaussians'))
          num.V1_Gaussians = normrnd(0,1,2,Nact*M);
       end
       if any(strcmpi(vars,'V2_Gaussians'))
          num.V2_Gaussians = normrnd(0,1,2,Nact*M);
       end

       % Create object:
       varargout = {RandomNumbersClass('numbers',num)};

   end
   function GenNumbersForMC(obj,ttm,type)
   % Description: Generates random numbers for one or several runs of
   % the Monte Carlo pricer and stores them in the obj.pricerSettings.random
   % property.
   %
   % Parameters:
   %    ttm:  [Nx1 real] Expiries we wish to price using Monte Carlo.
   %    type: [1x1 string (optional)] Options are
   %    
   %            o 'Gaussians': This is the recommended choice (and the
   %                           default). Here the underlying i.i.d.
   %                           standard normal random variables needed for 
   %                           running the Monte Carlo pricing algorithm
   %                           with the expiries given in 'ttm' are
   %                           generated and stored.
   %
   %            o 'Y':         This choice has a more narrow purpose. 
   %                           Here we pre-simulate the paths of the
   %                           process
   %
   %                            Y(t) :=  sqrt(2*alpha+1)
   %                                    *int_0^t (t-s)^{alpha} dW_1(s)
   %
   %                           and store those numbers for the Monte Carlo 
   %                           pricing algorithm to use. If the pricing 
   %                           algorithm also needs any processes that are 
   %                           measurable with respect to the path of the W_2(t) 
   %                           Brownian motion this will have to be simulated on 
   %                           the run as needed.
   % 


       settings = obj.pricerSettings;
       if ~isa(settings,'MonteCarloPricerSettingsClass')
           error(['rBergomiExtClass:GenNumbersForMC: ', ...
                  'Pricer settings must be an object of type', ...
                  ' ''MonteCarloPricerSettingsClass''.']);
       end
       
       anti = settings.price_estimation.antithetic;
       
       if ~exist('type','var') || isempty(type)
           type = 'Gaussians';
       end
       
       % Generate numbers:
       if strcmpi(type,'Gaussians')
            % Find the total number of steps needed:
            [~,~,nTotalSteps] = settings.GetNumSteps(ttm);
            
            % Set the processes required for price estimation:
            sim_variables = {'S'};
            
            % Generate random numbers:
            settings.random = obj.GenNumbersForSim(max(settings.N),...
                                                   max(nTotalSteps),...
                                                   sim_variables,anti);
                        
       elseif strcmpi(type,'Y')
           % Find the total number of steps needed but do so for each 
           % set of expiries having the same number of steps per year:
            [dtGrp,nPerYear,~] = settings.GetNumSteps(ttm);

            uniqDtGrp = unique(dtGrp);
            if settings.price_estimation.antithetic
                Nuse = max(settings.N)/2;
            else
                Nuse = max(settings.N);
            end

            cellArr = cell(size(uniqDtGrp,1),1);
            maxTByObj = NaN(size(uniqDtGrp,1),1);
            for i=1:size(uniqDtGrp,1)
                % Simulate processes for each expiry group:
                 rndobj = RandomNumbersClass();
                 rndobj.numbers = obj.Simulate(Nuse, ...
                        unique(nPerYear(dtGrp==uniqDtGrp(i))),...
                        max(ttm(dtGrp==uniqDtGrp(i))), ...
                        'outVars',{'Y'},...
                        'antithetic',false, ...
                        'scheme',settings.simulation.scheme,...
                        'timePoints','all_up_till');
                 rndobj.numbers.Y = rndobj.numbers.Y(:,1:end-1);
                 cellArr{i} = rndobj;
                 maxTByObj(i) = max(ttm(dtGrp==uniqDtGrp(i)));
            end
            obj.pricerSettings.randomMaxT = maxTByObj;
            obj.pricerSettings.random = cellArr;
       else
           error('rBergomiExtClass:GenNumbersForMC: Invalid input.');
       end
   end
   function paths = Simulate(obj, N, n, t, varargin)
    % Description: Simulates the extended rough Bergomi model assuming an 
    % initial asset price of 1 and zero drift.
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
    %           nearest grid point below. Important: 
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
    %                 then be returned for all the time points 
    %                       0 < 1/n < 2/n < ... < floor(t*n)/n
    %
    %               o 'all_up_till_excl_first': Here t must be scalar (M=1). 
    %                 Values will then be returned for all the time points 
    %                           1/n < 2/n < ... < floor(t*n)/n
    %
    %               o 'all_up_till_excl_last': Here t must be scalar (M=1). 
    %                 Values will then be returned for all the time points 
    %                       0 < 1/n < 2/n < ... < (floor(t*n) - 1)/n 
    %
    %   outVars:  [1xL] Cell array with names of the stochastic processes 
    %             you want outputted. Any subset of the variables: 
    %             'S', 'sig', 'V1', 'V2' can be chosen. These refer to the 
    %             processes S(t), sqrt(V(t)), V_1(t) and V_2(t) respectively 
    %             (see the class description). 
    %           
    %             Alternatively one is allowed to specify outVars = {'Y'}
    %             where 'Y' refers to the process
    %           
    %                Y(t) := sqrt(2*alpha+1)*int_0^t (t-s)^{alpha} dW_1(s).
    %
    %             This choice cannot currently be combined with any of 
    %             the others.
    %
    %             Finally, if outVars is left empty the default is to set
    %             outVars = {'S','sig'}.
    %
    %   antithetic: [1x1 logical] If true we return antithetic sample paths 
    %               as well (still N paths in total). It is then the first
    %               N/2 paths returned that will be the original paths and
    %               the bottom half (N/2) paths that will be antithetic. 
    %               The default is antithetic = false.
    %
    %   rndNumbers: [1x1 RandomNumbersClass] Object containing the
    %                random numbers that will be used for the simulation.
    %                If left empty we simulate them as needed.
    %
    %   scheme: [1x1 string] Simulation scheme to use. Currently the only
    %            allowed value is 'hybrid' (also the default) in which 
    %            case we use the hybrid scheme of (Bennedsen et al., 2017)
    %            with their kappa = 1.
    %
    % Output: 
    %   paths: [1x1 struct] Struct containing a field for each element in
    %          'outVars' as well as a field for the time points of the
    %          simulation.
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
       expectedOutVars = {'S','sig','V1','V2','Y'};
       
       if sum(ismember(outVars,expectedOutVars)) ~= size(outVars,2)
           error(['rBergomiClass:Simulate: Some output variables are ',...
                  'not supported']);
       end
       
       idxMember = ismember(expectedOutVars,outVars);
       
       if idxMember(end) && sum(idxMember(1:end-1) > 0)
           error(['rBergomiExtClass:Simulate: Currently when requesting ',...
             'the variable ''Y'' no other can be requested at the same time.']);
       elseif idxMember(end)
           return_Y = true;
       else
           return_Y = false;
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
       
       idxMaxT = sum(grid_pts <= t(end),1);

       % Simulate random variables:
       if isempty(rndNumbers)
            rndNumbers = obj.GenNumbersForSim(N,nStepsTotal,outVars,antithetic);
       end

       % Run simulation:
       switch scheme
           case 'hybrid'
               % Reparameterise:
               lambda = obj.rho*obj.eta;
               zeta = sqrt(obj.eta^2 - lambda^2);
               xi_unit = obj.xi.DeepCopy();
               xi_unit.values(:) = 1;
               
               % Find time indice to return
               idxRet = sum(grid_pts <= t',1);
               t_grid = grid_pts(1:idxMaxT);
               
               % Simulate each of the two factors:
               rnd_V1 = RandomNumbersClass();
               if isfield(rndNumbers.numbers,'V1_Gaussians')
                  rnd_V1.numbers.V_Gaussians = rndNumbers.numbers.V1_Gaussians;
               end
               if isfield(rndNumbers.numbers,'Y')
                   rnd_V1.numbers.Y = rndNumbers.numbers.Y;
               end
               
               if return_Y
                   ret_vars = outVars;
               else
                   ret_vars = {'V'};
               end
               
               paths_V1 = rBergomiClass.HybridScheme(obj.alpha+0.5,[],zeta,...
                            xi_unit,n,N,antithetic,t_grid,rnd_V1,ret_vars,...
                            obj.pricerSettings.simulation.conv_method,...
                            obj.pricerSettings.precision);
              
               if return_Y
                   paths = paths_V1;
                   return;
               end
                        
               if isfield(rndNumbers.numbers,'V2_Gaussians')
                   rnd_V2 = RandomNumbersClass();
                   rnd_V2.numbers.V_Gaussians = rndNumbers.numbers.V2_Gaussians;    
               else
                   rnd_V2 = [];
               end

               paths_V2 = rBergomiClass.HybridScheme(obj.beta+0.5,[],lambda,...
                            xi_unit,n,N,antithetic,t_grid,rnd_V2,{'V','dW1'},...
                            obj.pricerSettings.simulation.conv_method,...
                            obj.pricerSettings.precision);

               % Compute the variance process:
               V = bsxfun(@times,(paths_V1.V.*paths_V2.V),...
                                 (obj.xi.Eval(t_grid)).');
               sig_t = sqrt(V);

               % Compute the asset price process:
               dW2 = paths_V2.dW1(:,2:end);
               log_S_incr = sig_t(:,1:end-1) .* dW2 - 0.5.*V(:,1:end-1).*dt;
               log_S = cumsum(log_S_incr, 2);
               S = [ones(N,1),exp(log_S)];

               paths = struct;
               paths.t = t';
               if any(ismember(outVars,'S'));paths.S=S(:,idxRet);end
               if any(ismember(outVars,'sig'));paths.sig=sig_t(:,idxRet);end
               if any(ismember(outVars,'V1'));paths.V1=paths_V1.V(:,idxRet);end
               if any(ismember(outVars,'V2'));paths.V2=paths_V2.V(:,idxRet);end
               
           otherwise
               error('rBergomiExtClass:Simulate: Scheme is not supported.');
       end
   end   
end

end

