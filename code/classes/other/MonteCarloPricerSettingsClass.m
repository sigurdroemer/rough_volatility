classdef MonteCarloPricerSettingsClass < GenericSuperClass
% Description: Class to hold settings for a Monte Carlo based pricer as well 
% as some useful functionality in that regard.
%
% Properties:
%
%   o tn:                 [Mx1 real] Defines the expiry groups for which we
%                         require a different number of steps per year when
%                         simulating paths. The first group of expiries
%                         goes from 0 to obj.tn(1). The second from
%                         obj.tn(1) to obj.tn(2) etc. The number of steps
%                         per year for each group is then specified in obj.n.
%
%   o n:                  [Mx1 integer] The number of steps per year that
%                         should be used for each expiry group from obj.tn.
%
%   o N_vol_grp:          [Lx1 real or empty] If left empty then we use the
%                         same number of paths for all expiry groups no
%                         matter the volatility level. The number of paths is 
%                         then specified as a 1x1 integer under obj.N.
%                         Leaving the obj.N_vol_grp property non-empty
%                         is currently only supported for the rBergomiExtClass. 
%                         In that case we do the following: When the pricing 
%                         algorithm is called on a given set of expiries in an
%                         expiry-group (as specified by obj.tn) we then compute 
%                         the variance swap rates (in volatility terms) for all 
%                         expiries that need to be evaluated in that sub-group. 
%                         The number of paths is then chosen as follows: If the
%                         minimum variance swap rate (among all expiries in
%                         the sub-group) is between 0 and obj.N_vol_grp(1)
%                         then we use obj.N(1) paths to estimate prices. If
%                         the minimum rate is between obj.N_vol_grp(1) and
%                         obj.N_vol_grp(2) then we use obj.N(2) paths and
%                         so on. 
%                         REMARK: It is generally recommended to leave this
%                         parameter empty.
%
%   o N:                  [1x1 or Lx1 integer] Number of paths to be used unless 
%                         otherwise specified - see the nRepeat property. This
%                         number includes antithetic paths if these are enabled.
%                         Should be set to a general Lx1 integer vector only if
%                         the obj.N_vol_grp property is also a Lx1 vector (of real 
%                         numbers). Read the description for that property
%                         for more information.
%
%   o nRepeat:            [1x1 integer] Repeats the Monte Carlo pricing 
%                         algorithm this many times and takes the average of 
%                         prices across all repeats. The total number of paths 
%                         used is therefore obj.nRepeat*obj.N. Lowering N but 
%                         increasing nRepeat may be useful if your system have 
%                         limited  memory. However be aware that certain
%                         price estimation methods get increasingly biased
%                         as obj.N gets smaller even if obj.nRepeat gets
%                         comparatively larger. The same goes for
%                         estimating the standard errors which are
%                         aggregated as
%
%                                 average standard error 
%                                 over obj.nRepeat runs
%                                           / 
%                                  sqrt(obj.nRepeat)  
%
%   o simulation:         [1x1 struct] Struct containing settings on 
%                         how paths should be simulated. Must contian the
%                         following fields:
%
%                           o scheme: 
%                               [1x1 string] Specifies which scheme to use. 
%                               Currently the only option is 'hybrid' in which 
%                               case the hybrid scheme of 
%                               (Bennedsen et al., 2017) is used to simulate 
%                               any stochastic Volterra processes. A log-euler 
%                               scheme is then used for the asset price if 
%                               needed.
%                           
%                           o conv_method: 
%                               [1x1 string] Specifies which method to use to 
%                               compute the convolutions for the hybrid scheme. 
%                               Options are 'fft', 'conv2' and 'optimal'. The
%                               latter is recommended as it attempts to choose 
%                               the fastest method depending on the number
%                               of steps that needs to be simulated.
%
%   o price_estimation:   [1x1 struct] Struct specifying how the price
%                         estimation is to be done. Most of the techniques
%                         are covered in (McCrickerd and Pakkanen, 2018) for
%                         the rough Bergomi model although the techniques
%                         apply more generally. Below we briefly cover each of 
%                         the fields that should be specified. You can consult 
%                         the MonteCarloEstimation method of the 
%                         PricingModelClass for even more details.
%
%                           o control_variate: 
%                               [1x1 string] Specifies which control variate to 
%                               use. Options are 'asset_price', 'timer_option' 
%                               and 'none'.
%                       
%                           o antithetic: 
%                               [1x1 logical] If true half of the paths used 
%                               will be antithetic.
%
%                           o conditional_monte_carlo: 
%                               [1x1 logical] If true the price estimation
%                               will use a conditional Monte Carlo estimator.
%
%                           o option_type: 
%                               [1x1 string] Specifies which option type should 
%                               be used to estimate prices. Possible choices 
%                               are 'call', 'put' and 'otm' (out-of-the-money) 
%                               options. The latter is recommended.
%
%
%   o precision:          [1x1 string] Specifies in which precision the
%                         most heavy computations should be done. Options are 
%                         'double' and 'single'.
%
%   o random:             [(depends)] Stores any random variables needed for 
%                         simulating paths. Allows one to reuse the random 
%                         numbers if so desired. Generally not to be used
%                         by an end-user but rather is mostly intended to be 
%                         used internally by a pricing model to temporarily
%                         store random numbers.
%
%   o randomMaxT:         [(depends)] Not intended to be used by an
%                         end-user. Is used internally by a pricing model
%                         to properly interpret the obj.random property.
%
% References:
%   o Bennedsen, M., Lunde, A. and Pakkanen, M.S., Hybrid scheme for Brownian 
%     semistationary procesess. Finance and Stochastics, 2017, 21(4), 931-965.
%   o McCrickerd, R. and Pakkanen, M.S., Turbocharging Monte Carlo pricing for 
%     the rough Bergomi model. Quantitative Finance, 2018, 18(11), 1877-1886.
%
    
properties
    tn
    n
    N_vol_grp
    N
    nRepeat    
    simulation
    price_estimation
    precision
    random
    randomMaxT
end
    
methods
    function obj = MonteCarloPricerSettingsClass(varargin)
    % Description: Constructor. 
    % 
    % Parameters: Inputs must be given in name-value pairs corresponding to 
    % the object properties. If some properties are not set they may be
    % set to some default values.
    %
    % Output:
    %   [1x1 MonteCarloPricerSettingsClass] The object.
    %

        obj.ParseConstructorInputs(varargin{:});

        % Default values:
        if isempty(obj.tn) && size(obj.n,1)==1;obj.tn=Inf;end
        if isempty(obj.nRepeat) ;obj.nRepeat=1;end
        if isempty(obj.precision);obj.precision = 'double';end

    end
    function [idxGrp,nPerYear,nActual] = GetNumSteps(obj,T)
    % Description: Returns the number of simulation steps needed to price 
    % each of the input expiries (and some other information).
    %
    % Parameters:
    %   T: [Nx1 real] Expiries to be priced.
    %
    % Output:
    %   idxGrp:   [Nx1 integer] Vector with indices grouping the expiries
    %             according to how many steps per year are needed.
    %
    %   nPerYear: [Nx1 integer] The number of steps per year required to
    %             simulate to each respective maturity.
    %
    %   nActual:  [Nx1 integer] The actual number of steps required to
    %             simulate to each respective maturity.
    %

       idxGrp = size(obj.tn,1) - sum(bsxfun(@(x,y)(x<=y),T,obj.tn'),2) + 1;
       nPerYear = obj.n(idxGrp);
       nActual = double(int64(T.*obj.n(idxGrp)));
       
    end
    function ClearNumbers(obj)
    % Description: Clears random numbers.
        obj.random = [];
    end    
end

end

