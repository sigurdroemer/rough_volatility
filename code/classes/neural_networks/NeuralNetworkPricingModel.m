classdef NeuralNetworkPricingModel
% Description: Interface for a pricing model approximated by one or several 
% neural networks.
%
% Properties:
%   k:       [Nx1 real] Log-moneyness grid values
%   T:       [Nx1 real] Expiry grid values.
%   nn:      [Lx1 cell] Neural networks to compute implied volatilities.
%            Each element must therefore be of the class NeuralNetwork.
%   lb:      [Mx1 real] Lower bound of allowed inputs to the neural networks.
%   ub:      [Mx1 real] Upper bound of allowed inputs to the neural networks.
%   in_idx:  [Lx1 cell] The i'th element is a vector of integers specifying
%            which indices of the total input vector should be inputted to
%            each particular neural network.
%   out_idx: [Lx1 cell] The i'th element is a vector of integers specifying
%            which indices of the total output vector that comes from each
%            particular neural network.
%   label:   [1x1 string] A label for the object. Does not influence the behaviour.
%

properties
    k
    T
    nn
    in_idx
    out_idx
    lb
    ub
    label
end
    
methods
function obj = NeuralNetworkPricingModel(nn,in_idx,out_idx,k,T,lb,ub,label)
% Description: Constructor.
%
% Parameters: Inputs correspond exactly to the object properties of the same 
% name. See the class description for more information.
% 
    obj.nn = nn;
    obj.in_idx = in_idx;
    obj.out_idx = out_idx;
    obj.k = k;
    obj.T = T;
    obj.lb = lb;
    obj.ub = ub;
    obj.label = label;
end
function preCompInfo = PerformPrecomputations(obj,kq,Tq)
% Description: Precomputes various information which is useful when performing 
% repeated interpolation of a neural network based implied volatility surface 
% using natural cubic splines in the (log)moneyness dimension and linear 
% interpolation in the expiry dimension (in terms of the total variance).
% 
% Important remark: We do assume that the (kq,Tq) vectors are sorted
% by expiration first and next by the log-moneyness. We also assume that 
% all contracts are within the supported domain of the neural network(s).
%
% Parameters:
%   kq: [Nx1 real] Log-moneyness values to evaluate the surface in.
%   Tq: [Nx1 real] Expiry values to evaluate the surface in. 
%
% Output: A [1x7 cell] array containing the elements described below where
% we add the variable names from the code:
%   Tgrps:     [2xM real] Upper and lower time points of each interpolation group.
%   idxQ:      [2xM integer] Index range of query points corresponding to each 
%              interpolation group.
%   idxBelow:  [2xM integer] Index range of grid points corresponding to the 
%              lower expiration slice.
%   idxAbove:  [2xM integer] Index range of grid points corresponding to the 
%              upper expiration slice.
%   frac:      [1xM cell] The i'th entry contains a vector of size 1xL(i) where 
%              L(i) is the number of query points corresponding to the i'th 
%              interpolation group and where the entries of this vector are the 
%              fractions of the interval between the lower and upper expiration 
%              that the query points sit.
%   Sl:        [1xM cell] Each element is a cell array of size [1x6] containing 
%              precomputed information for interpolating the lower expiration 
%              slice. See the function 'PrecomputeSpline' for the meaning of 
%              each output.
%   Su:        [1xM cell] Same as for Sl except for the upper expiration slice.
%

    % Find the unique expiry values (for both grid and query points)
    uniqT = unique(obj.T);
    uniqT_query = unique(Tq);

    % Initialize outputs:
    [Su,Sl,frac] = deal({[]});
    [idxAbove,idxBelow,idxQ,Tgrps] = deal([]);

    % Define vector keeping track of the query expirations that
    % remain to be processed:
    Tqueue = uniqT_query;

    % Initialize variables for the loop:
    i = 1;
    atUpperBound = false;

    % Loop over all (unique) query expirations (from the smallest 
    % to the largest):
    while ~isempty(Tqueue)

        % Find expiration (in grid) just above the (smallest 
        % remaining) query expiration:
        idxJustAbove = find(uniqT > min(Tqueue),1,'first');

        % Note it down if the query expiration is at the largest 
        % expiration in the grid:
        if isempty(idxJustAbove)
            idxJustAbove = size(uniqT,1);
            atUpperBound = true;
        end

        % Find expiration (in grid) just below the (smallest 
        % remaining) query expiration:
        idxJustBelow = idxJustAbove - 1;

        % Find the associated expiration values:
        Tabove = uniqT(idxJustAbove);
        Tbelow = uniqT(idxJustBelow);

        % Find query points between these two grid expirations 
        % (rather the indices of these):
        if ~atUpperBound
            idx_in = Tq >= Tbelow & Tq < Tabove;
        else
            idx_in = Tq >= Tbelow & Tq <= Tabove;
        end

        % Collect the expirations of the two slices to interpolate 
        % between:
        Tgrps = [Tgrps,[Tbelow;Tabove]];

        % Compute indices in grid

        % Get grid points (indices) at the upper and lower 
        % expiration slices and collect them in a matrix:
        idx_grid_just_above = obj.T == Tabove;
        idx_grid_just_below = obj.T == Tbelow;
        idx_grid_new = [find(idx_grid_just_above,1,'first');...
                        find(idx_grid_just_above,1,'last')];
        idxAbove = [idxAbove,idx_grid_new];
        idx_grid_new = [find(idx_grid_just_below,1,'first');...
                        find(idx_grid_just_below,1,'last')];
        idxBelow = [idxBelow,idx_grid_new];

        % Collect indices of query points between the two slices:
        idxQ_new = [find(idx_in,1,'first');find(idx_in,1,'last')];
        idxQ = [idxQ,idxQ_new];

        % Collect the (expiration) time fraction of the query 
        % points starting at zero from the lower slice going to 1 
        % (or 100%) at the upper slice:
        frac{i} = (Tq(idx_in) - Tbelow)./(Tabove-Tbelow);

        % Collect precomputed variables for the splines at both the  
        % lower and upper slice:
        Su{i} = obj.PrecomputeSpline(...
                obj.k(idx_grid_just_above),kq(idx_in));
        Sl{i} = obj.PrecomputeSpline(...
                obj.k(idx_grid_just_below),kq(idx_in));

        % Remove the query expiration just processed from the 
        % queue:
        if ~atUpperBound
            Tqueue = Tqueue(Tqueue >= Tabove);
        else
            Tqueue = Tqueue(Tqueue > Tabove);
        end

        i = i + 1;
    end

    preCompInfo = {Tgrps,idxQ,idxBelow,idxAbove,frac,Su,Sl};

end
function s = PrecomputeSpline(~,x,xq)
% Description: Precomputes a lot of information to be used for repeated 
% fitting and evaluation of a single natural cubic spline. The basic 
% assumptions is that the grid points for fitting the spline are fixed and 
% that the points to evaluate the spline in are also fixed.
%
% To make sense of the code (and the output) the situation is described in
% more detail below. Most of the notation follows section 2.3 from 
% (Gautschi, 2011).
%
% Let (x,f) := (x_i,f_i) i=1,...,n denote the points to interpolate with
% a natural cubic spline and let xq_i i=1,...,m be the points at which we
% wish to evaluate the spline after fitting it to (x,f).
%
% Without loss of generality we assume the x-vector is sorted in ascending
% order.
% 
% A natural cubic spline S(x) over the grid points (x_i; i=1,...,n) can then
% be written as
%
%   S(x) = P_i(x) when x_{i} <= x <= x_{i+1}
%
% where for i=1,2,...,n-1
%
%   P_i(x) = c_{i,0} + c_{i,1}*(x-x_i) + c_{i,2}*(x-x_i)^2 + c_{i,3}*(x-x_i)^3
%
% for x_i <= x <= x_{i+1}.
%
% In the above, the coefficients (c_{i,0},c_{i,1},c_{i,2},c_{i,3}) i=1,...,n-1 
% must be chosen so S(x) is two times continuously differentiable on [x_1,x_n] 
% and such that S''(x_1) = S''(x_n) = 0. Following (Gautschi, 2011), 
% section 2.3.4, one has
%
%   c_{i,0} = f_i
%   c_{i,1} = m_i
%   c_{i,2} = (((f_{i+1} - f_i)/(x_{i+1} - x_i)) - m_i)/delta_i - c_{i,3}*delta_i
%   c_{i,3} = (m_{i+1} + m_i - 2*(f_{i+1} - f_i)/(x_{i+1} - x_i))/(delta_i^2)
%
%   delta_i = x_{i+1} - x_i
%
% for i=1,2,...,n-1 where the m-vector solves the n-dimensional system of equations
%
%   Am = d
%
% with
%
%      |b_1    c_1                                          |
%      |a_1    b_2     c_2                    0             |
%      |       a_2     b_3     c_3                          |
% A:=  |                ...     ...     ...                 |
%      |           0             ...     ...     c_{n-1}    |
%      |                                a_{n-1}  b_n        |
%
%
% and
%
%   b_1     = 2
%   b_i     = 2*(delta_{i-1} + delta_i) , i=2,...,n-1
%   b_{n}   = 2
%
%   a_i     = delta_{i+1}, i=1,...,n-2
%   a_{n-1} = 1
%
%   c_1     = 1
%   c_i     = delta_{i-1}, i=2,...,n-1
%
% and
%
%   d_1 = 3*(f_2-f_1)/(x_2-x_1)
%   d_i = 3*{delta_i*((f_i-f_{i-1})/(x_i-x_{i-1})) 
%         + delta_{i-1}*((f_{i+1}-f_i)/(x_{i+1}-x_i))}
%   d_n = 3*(f_n-f_{n-1})/(x_n-x_{n-1})
%
% In the code we assume x_{i+1} - x_i is constant, i.e. the grid points are
% equidistant. In addition we scale the x-values so x_i = i for i=1,...,n 
% and delta_i = 1 for i=1,...,n-1. Under these assumptions a number of things 
% simplfy:
%
%   b = [2,4,...,4,2]'
%   a = [1,1,...,1,1]'
%   c = [1,1,...,1,1]'
%
%   d_1 = 3*(f_2 - f_1)
%   d_i = 3*(f_{i+1} - f_{i-1}) for i=2,...,n-1
%   d_n = 3*(f_n - f_{n-1}).
%
% In addition,
%
%   S(x) =  f_i     * { 1 - 3*(x-x_i)^2 + 2*(x-x_i)^3      }
%           f_{i+1} * { 3*(x-x_i)^2 - 2*(x-x_i)^3          }
%           m_i     * { (x-x_i) - 2*(x-x_i)^2 + (x-x_i)^3  }
%           m_{i+1} * { -(x-x_i)^2 + (x-x_i)^3             }
%
%        =: f_i * w_1 + f_{i+1} * w_2 + m_i * w_3 + m_{i+1} * w_4
%
% whenever x_i <= x <= x_{i+1} and where we have defined the coefficients
% w_1,...,w_4 appropriately.
%
% Parameters:
%   x:   [nx1 real] Grid points for natural cubic spline to interpolate 
%        between. We assume them equidistant and sorted in ascending order.
%   xq:  [mx1 real] Query points to evaluate the natural cubic spline in. 
%        We assume that x_1 <= min(xq) <= max(xq) <= x_n.
%
% Output: 
%   s: [1x6 cell] Cell array containing the following elements:
%
%           o Ainv:       [nxn real] The A matrix inverted.
%           o w1:         [1x1 real] See the description.
%           o w2:         [1x1 real] See the description.
%           o w3:         [1x1 real] See the description.
%           o w4:         [1x1 real] See the description.
%           o xq_floored: [mx1 real] xq values appropriately converted to
%                         the range [1,n] and then truncated down to the
%                         nearest integer.
%
% References:
%   o  Gautschi, Walter, Numerical Analysis, 2nd ed., 2011, Birkhauser.
%

    % Compute A-matrix and invert it:
    n = size(x,1);
    A = toeplitz([4 1 zeros(1,n-2)]);
    A(1,1) = 2;
    A(end,end) = 2;
    Ainv = inv(A);

    % Convert xq to the integer range:
    dx = x(2) - x(1);minx = x(1);
    xq_adj = (xq - minx)./dx + 1;
    xq_floored = floor(xq_adj); % points to grid point just below
    xq_floored(xq_floored==n) = n - 1;
    
    x_diff = xq_adj - xq_floored;
    w1 = 1 - 3*x_diff.^2 + 2*x_diff.^3;
    w2 = 3*x_diff.^2 - 2*x_diff.^3;
    w3 = x_diff - 2*x_diff.^2 + x_diff.^3;
    w4 = -x_diff.^2 + x_diff.^3;

    % Store information in cell-array:
    s = {Ainv,w1,w2,w3,w4,xq_floored};
end
function ivs = InterpolateNeuralNetwork(obj,ivGrid,Tgrps,idxQ,idxBelow,...
                                        idxAbove,frac,Su,Sl,k,T)
% Description: Evaluates and interpolates a neural network representing the 
% implied volatility surface. This function requires extra pre-prepared inputs 
% which in turn speeds up repeated evaluation.
%
% Parameters:
%   ivGrid:     [Nx1 real] Implied volatilities at grid points.
%   Tgrps:      [2xM real] Upper and lower time points of each interpolation 
%               group.
%   idxQ:       [2xM integer] Index range of query points corresponding to each 
%               interpolation group.
%   idxBelow:   [2xM integer] Index range of grid points corresponding to the 
%               lower expiration slice.
%   idxAbove:   [2xM integer] Index range of grid points corresponding to the 
%               upper expiration slice.
%   frac:       [1xM cell] The i'th entry contains a vector of size 1xL(i) 
%               where L(i) is the number of query points corresponding to the 
%               i'th interpolation group and where the entries of this vector 
%               are the fractions of the interval between the lower and upper 
%               expiration that the query points sit.
%   Sl:         [1xM cell] Each element is a cell array of size [1x9] containing 
%               precomputed information for interpolating the lower expiration 
%               slice. See the class method 'PrecomputeSpline' for the meaning of 
%               each output.
%   Su:         [1xM cell] Same as for Sl except for the upper expiration slice.
%
% Output:
%   ivs: [Nx1 real] 
%

    ivs = NaN(size(k));

    % Loop over interpolation groups, evaluate splines, then
    % interpolate in the time-dimension:
    for i=1:size(Su,2)
        ivs(idxQ(1,i):idxQ(2,i),:) = sqrt(...
        (frac{i}.*Tgrps(2,i)...
        .*obj.EvalSpline(Su{i},...
                    ivGrid(idxAbove(1,i):idxAbove(2,i),:)).^2 ...
        + (1 - frac{i}).*Tgrps(1,i)...
        .*obj.EvalSpline(Sl{i},...
                    ivGrid(idxBelow(1,i):idxBelow(2,i),:)).^2)...
        ./T(idxQ(1,i):idxQ(2,i)));
    end

end
function yq = EvalSpline(~,s,f)
% Description: Evaluates a natural cubic spline where a number of
% precomputations have already been done. 
%
% Parameters:
%   s:  [1x6 cell] Exactly the output of the class method 
%       obj.PrecomputeSpline. See that method for more information.
%   f:  [Nx1 real] Values at knots of spline.
%
% Output:
%   fq: [Mx1 real] Value of spline at query points.
%

    d = NaN(size(f));
    d(2:end-1) = 3*(f(3:end) - f(1:end-2));
    d(1) = 3*(f(2)-f(1));
    d(end) = 3*(f(end)-f(end-1));
    
    m = s{1}*d;
    
    yq = s{2}.*f(s{6}) + s{3}.*f(s{6}+1) ...
         + s{4}.*m(s{6}) + s{5}.*m(s{6}+1);

end
function [kq_adj,Tq_adj] = FilterContracts(obj,kq,Tq,cartProd)
% Description: Removes input contracts that are not within the neural network 
% domain.
%
% Parameters:
%   kq:       [nx1 real] Log-moneyness.
%   Tq:       [mx1 real] Expiries.
%   cartProd: [1x1 logical] If true we interpret the (kq,Tq) vectors as 
%             producing a cartesian product of contracts. If false we do not 
%             in which case we require n = m. If empty we set it to true if 
%             n <> m and false otherwise.
%
% Output:
%   kq_adj:   [Nx1 real] Log-moneyness values after pruning.
%   Tq_adj:   [Nx1 real] Expiries after pruning.

    n = size(kq,1);
    m = size(Tq,1);

    if ~exist('cartProd','var') || isempty(cartProd)
        if n == m
            cartProd = false;
        else
            cartProd = true;
        end
    end

    if ~cartProd && n ~= m
        error(['NeuralNetworkPricingModel:FilterContracts:',...
                ' When cartesian product is used the length',...
                ' of the input vectors must be the same.']);        
    elseif cartProd
        % Create vectors.
        kq = repmat(kq,m,1);
        Tq = repmat(Tq,1,n)';
        Tq = Tq(:);
    end

    idxValid = obj.AreContractsInNetworkDomain(kq,Tq);
    kq_adj = kq(idxValid);
    Tq_adj = Tq(idxValid);

end
function idx = AreContractsInNetworkDomain(obj,kq,Tq)
% Description: Checks if a set of contracts are within the
% domain of the neural networks and thus can be interpolated.
%
% Parameters:
%   kq: [nx1 real] Log-moneyness
%   Tq: [nx1 real] Expiries
%
% Output:
%   idx: [nx1 logical] Element is true if contract is within
%        the domain, otherwise false.
%

    if size(kq,1) ~= size(Tq,1) || size(kq,2) ~= 1 ...
            || size(Tq,2) ~= 1
        error(['NeuralNetworkPricingModel: ', ...
               'AreContractsInNetworkDomain: Invalid input ', ...
               'dimensions']); 
    end
    idx = true(size(kq));
    uniqT = unique(Tq);uniqTGrid = unique(obj.T);
    minT = min(obj.T);maxT = max(obj.T);
    for i=1:size(uniqT,1)
        idxT = Tq == uniqT(i);

        if uniqT(i) > maxT || uniqT(i) < minT
            idx(idxT) = false;
            continue;
        end

        % Find expiration (in grid) just above the (smallest 
        % remaining) query expiration:
        idxAbove = find(uniqTGrid > uniqT(i),1,'first');

        if isempty(idxAbove)
            idxAbove = size(uniqTGrid,1);
        end

        % Find expiration (in grid) just below the (smallest 
        % remaining) query expiration:
        idxBelow = idxAbove - 1;                

        % Check moneyness values:
        idxGridBelow = obj.T == uniqTGrid(idxBelow);
        idxGridAbove = obj.T == uniqTGrid(idxAbove);

        idx(idxT) = kq(idxT) >= max(min(obj.k(idxGridBelow)),...
                                         min(obj.k(idxGridAbove))) ...
                         & kq(idxT) <= min(max(obj.k(idxGridBelow)),...
                                         max(obj.k(idxGridAbove)));

    end

end
function [ivs, kq, Tq] = Eval(obj,par,kq,Tq,cartProd,preCompInfo,skipChecks)
% Description: Returns Black-Scholes implied volatilities of the (neural 
% network approximated) pricing model.
%
% Parameters:
%   par:            [Px1 real] Input parameters.
%   kq:             [Nx1 real] Log-moneyness.
%   Tq:             [Mx1 real] Expiries.
%   cartProd:       [1x1 logical] If true we return implied volatilities in 
%                   the entire cartesian product of the kq and Tq vectors. If
%                   false we do it row-wise in which case we require N = M. 
%                   Default (if left empty) is to use cartesian product if 
%                   N <> M and otherwise not.
%   preCompInfo:    [1x1 cell (optional)] Cell array containing various 
%                   information used for fast repeated evaluation at the same 
%                   input points (kq,Tq). Must be obtained from making the 
%                   following call: preCompInfo = obj.GetPrecomputations(kq,Tq)
%   skipChecks:     [1x1 logical (optional)] If true we skip most validation 
%                   checks. Will speed up repeated computation for the same
%                   input contracts by some. Default is false. Should be used 
%                   with care. It is the user's responsibility to ensure the 
%                   following things:
%                       o cartProd = false
%                       o [Tq,kq] must be sorted row-wise by expiration first 
%                         and then moneyness.
%                       o All input contracts must be within the domain of
%                         the model. You can use the obj.AreContractsInNetworkDomain
%                         method to check this.
%                       o The input parameter par must satisfy the parameter 
%                         bounds of the pricing model as set in the object 
%                         properties obj.lb and obj.ub.
%                       o preCompInfo is not empty.
%
%                   Warning: The above assumptions will not be validated in
%                   the code.
%
% Output:
%   ivs:    [Nx1 or NxM real] Implied volatilities.
%   kq:     [Nx1 or NxM real] Log-moneyness.
%   Tq:     [Nx1 or NxM real] Expiries.
% 

    n = size(kq,1);
    m = size(Tq,1);

    if ~exist('skipChecks','var') || isempty(skipChecks)
        skipChecks = false;
    end
    
    if ~skipChecks
        if any(par < obj.lb | par > obj.ub )
           error(['NeuralNetworkPricingModel:Eval: Input parameter',...
                  ' is outside range of neural network.']);
        end

        if ~exist('cartProd','var') || isempty(cartProd)
            if n == m
                cartProd = false;
            else
                cartProd = true;
            end
        end

        % Check the input size:
        if (n ~= m && ~cartProd) || size(kq,2) > 1 || size(Tq,2) > 1
           error(['NeuralNetworkPricingModel:Eval: Invalid input',...
                  ' dimensions.']);
        end

        if cartProd
            % Construct grid as vectors:
            kq_vec = repmat(kq,m,1);
            Tq_vec = reshape(repmat(Tq,1,n)',n*m,1);
        else
            kq_vec = kq;
            Tq_vec = Tq;
        end

        % Sort inputs:
        [tmp, idxSort] = sortrows([Tq_vec,kq_vec]);
        Tq_vec = tmp(:,1);
        kq_vec = tmp(:,2);

        idxValid = obj.AreContractsInNetworkDomain(kq_vec,Tq_vec);
        if any(~idxValid)
            warning(['NeuralNetworkPricingModel:Eval: ', ...
                    'Some input contracts were outside grid. ', ...
                    'Values will be returned as NaN.']);
            if ~any(idxValid)
                ivs = NaN(size(kq_vec,1),1);
                return;
            end
        end

        % Do precomputations if not inputted directly:
        if ~exist('preCompInfo','var') || isempty(preCompInfo)
          preCompInfo = obj.PerformPrecomputations(kq_vec(idxValid),...
                                                   Tq_vec(idxValid));
        end

    else
        kq_vec = kq;
        Tq_vec = Tq;
        idxValid = true(size(kq));
    end
    
    % Evaluate the neural network(s):
    ivs_grid = NaN(size(obj.k));
    for i=1:size(obj.nn,1)
        ivs_grid(obj.out_idx{i}) = obj.nn{i}.Eval(par(obj.in_idx{i}));
    end
    
    % Interpolate:
    ivs = NaN(size(kq_vec,1),1);
    ivs(idxValid) = obj.InterpolateNeuralNetwork(ivs_grid,...
                                                 preCompInfo{1},...
                                                 preCompInfo{2},...
                                                 preCompInfo{3},...
                                                 preCompInfo{4},...
                                                 preCompInfo{5},...
                                                 preCompInfo{6},...
                                                 preCompInfo{7},...
                                                 kq_vec(idxValid),...
                                                 Tq_vec(idxValid));

    if skipChecks
        return;
    end
    
    if cartProd
        % Undo sorting and reshape:
        ivs = reshape(ivs(idxSort),n,m);
        kq = repmat(kq,1,m);
        Tq = repmat(Tq,1,n)';
    else
        % Undo sorting:
        ivs(idxSort) = ivs;
    end

end
end

end

