function paths = HybridScheme(H,rho,eta,xi,n,N,anti,retPts,rndNumbers,...
                              outVars,convMethod,prec)
% Description: Simulates the rough Bergomi model using the hybrid scheme of
% (Bennedsen et al., 2017) with their kappa = 1. We simulate under the 
% assumption of an initial asset price of 1 and no drift (i.e. zero interest 
% rates and dividends).
%
% Let us briefly explain the model. Let first W_1 and W_perp be two independent 
% Brownian motions and define for a -1 <= rho <= 1 a new Brownian motion 
% W_2(t) := rho*W_1(t) + sqrt(1-rho^2)*W_perp(t). 
%
% The model for the asset price S(t) can then be written as
%
%   dS(t) = S(t)sqrt(V(t))dW_2(t)
%         
%         = S(t)sqrt(V(t))d(rho*W_1(t) + sqrt(1-rho^2)*W_perp(t))
%
% where
%
%   V(t) = xi(t)*exp(eta*Y(t) - (eta^2/2)*t^(2*alpha+1))
%
% with xi(t) being a deterministic function (the initial forward variance
% curve), eta > 0, alpha = H - 0.5, 0 < H < 1/2 and where
%
%   Y(t) = sqrt(2*alpha + 1)*int_0^t (t-s)^{alpha}dW_1(s).
%
% To introduce additional notation define also
%
% S_1(t)   := exp(rho*int_0^t sqrt(V(s))dW_1(s)-0.5*rho^2*int_0^t V(s)ds)
% S_perp(t):= exp(sqrt(1-rho^2)*int_0^t sqrt(V(s))dW_perp(s)
%                 -0.5*(1-rho^2)*int_0^t V(s) ds).
%
% Then we can also write
%
%   S(t) = S_1(t)*S_perp(t).
% 
% Remarks:
%   o The code is (partly) optimized for performance and memory allocations. 
%     It may be somewhat less readable because of this.
%
%   o The function does not use any properties from the class 'rBergomiClass' 
%     and is thus independent of the specific instance of that object.
%
% Parameters: 
%   H:              [1x1 real] Hurst exponent.
%
%   rho:            [1x1 real] Correlation parameter.
%
%   eta:            [1x1 real] Volatility-of-volatility parameter.
%
%   xi:             [1x1 CurveClass] The (initial) forward variance curve.  
%
%   n:              [1x1 integer] Number of steps per year.
%
%   N:              [1x1 integer] Number of distinct paths to simulate. This 
%                   includes any antithetic paths. If anti = true then N must 
%                   be divisible by 2.
%
%   anti:           [1x1 logical] If set to true half of the paths will be 
%                   antithetic.
%
%   retPts:         [Mx1 real] Time points to return simulated processes on.
%                   If the points do not fit into the grid
%                   0 < 1/n < 2/n < ... < floor(max(retPts)*n)/n
%                   they will be truncated to the nearest grid point.
%
%   outVars:        [1xL cell] Cell array with names (strings) of the stochastic  
%                   processes to be outputted. Options are:
%                      o 'S':           Returns S(t).
%                      o 'V':           Returns V(t).
%                      o 'sig':         Returns sqrt(V(t)).
%                      o 'Y':           Returns Y(t).
%                      o 'dW1':         Returns dW_1(t).
%                      o 'QV':          Returns int_0^t V(s) ds.
%                      o 'int_sig_dW1': Returns int_0^t sqrt(V(s)) dW_1(s).
%                      o 'S1':          Returns S_1(t).
%                      o 'log_S1':      Returns log(S_1(t)).
%
%                   Remark: The processes S(t), S_1(t) and log(S_1(t)) are
%                   all simulated using a log-euler scheme.
%
%   rndNumbers:     [1x1 RandomNumbersClass or empty] Contains random numbers 
%                   to be used by the simulation scheme. If the parameter
%                   is left empty all necessary random variables are
%                   simulated automatically in the code. However if an
%                   appropriate RandomNumbersClass object is inputted some
%                   (or all) of the random numbers can be given as inputs.
%                   How this works is explained in the next paragraphs
%                   although it should be stressed that one should be
%                   careful to ensure that the inputs are given correctly
%                   so the computed paths are also correct.
%
%                   Firstly, to simulate all processes available except the 
%                   S(t) process this object must contain the field 
%                   rndNumbers.numbers.V_Gaussians of independent standard 
%                   Gaussians random variables. This field should be a matrix
%                   with the shape 2 x (nIntervals*Nindep) where 
%                   nIntervals = floor(n*max(retPts)) is the number of time 
%                   intervals needed for the simulation and where Nindep = N/2  
%                   if antithetic paths are used and Nindep = N if they are not. 
%                   The field can be left empty. In this case the numbers will 
%                   be simulated automatically in the code. 
%
%                   Secondly, to simulate S(t) the field 
%                   rndNumbers.numbers.Wperp_Gaussians should also be specified. 
%                   This field should then be a matrix with dimensions 
%                   Nindep x nIntervals and again consist of independent 
%                   standard Gaussian random variables. It too can be left 
%                   empty in which case these numbers will also be simulated 
%                   automatically when needed. 
%
%                   Thirdly, instead of specifying the i.i.d. standard normals 
%                   in rndNumbers.numbers.V_Gaussians it is possible to instead 
%                   specify the fields rndNumbers.numbers.dW1 and 
%                   rndNumbers.numbers.Y with already simulated increments of 
%                   the Brownian motion W_1(t) and paths of the process Y(t). 
%                   The dimension of these matrices should then be 
%                   Nindep x nIntervals and consist of the values 
%                   W(i/n) - W((i-1)/n) and Y(i/n) for i=1,2,...,nIntervals.
%
%                   Finally if only V(t) (and/or Y(t) for that matter) is
%                   requested as outputs then it is also possible to only
%                   specify pre-simulated paths of the Y(t) process under 
%                   rndNumbers.numbers.Y as already explained.
%
%   convMethod:     [1x1 string] Computational method for computing the 
%                   convolution of two vectors. Options are 'optimal', 
%                   'fft' and 'conv2'. The recommended option is 'optimal'
%                   which selects either 'fft' or 'conv2' based on the
%                   length of the input vectors and that in an optimal way.
%
%   prec:           [1x1 string] Precision to be used. Options are 'single'
%                   and 'double'.
%
% Output: 
%   paths: [1x1 struct] A struct containing a field 't' with the time points 
%          and in addition fields for each element in the 'outVars' parameter.
%
% References: 
%   o Bennedsen, M., Lunde, A. and Pakkanen, M.S., Hybrid scheme for Brownian 
%     semistationary procesess. Finance and Stochastics, 2017, 21(4), 931-965.
% 

    %% Initialization and input validation
    % Compute some useful constants:
    alpha = H - 0.5;
    gamm = - alpha;
    dt = 1 / n;
    
    % Adjust (if necessary) the required output time points to fit into the grid.
    grid_pts_temp = (1/n)*(0:floor(max(retPts)*n + 1))';
    idxRet = sum(grid_pts_temp <= retPts',1)';
    t = dt*(0:max(idxRet - 1))';
    retPtsAdj = t(idxRet);
    
    if strcmpi(prec,'single')
        t = single(t);
        xi_eval = single(xi.Eval(t));
    else
        xi_eval = xi.Eval(t);
    end

    nTimePts = size(t, 1);
    nIntervals = nTimePts - 1;

    % Initialize output struct:
    paths = struct;
    paths.t = retPtsAdj';
    for i=1:size(outVars,2)
        paths.(outVars{i}) = [];
    end

    % Manage some things for antithetic sampling:
    idxAnti = false(N,1);
    if anti
        if mod(N,2) ~= 0
            error(['rBergomiClass:HybridScheme: With antithetic ',...
                   'sampling enabled the number of paths must be ',...
                   'divisible by 2.']);
        end
        idxAnti(N/2+1:end) = true;
        Nindep = N / 2;
    else
        Nindep = N;
    end

    %% Simulation algorithm
    if ~isempty(rndNumbers) && isfield(rndNumbers.numbers,'Y') ...
                            && isfield(rndNumbers.numbers,'dW1')
        % Unpack random numbers and skip their simulation.
        Y = rndNumbers.numbers.Y(1:Nindep,:);
        dW1 = rndNumbers.numbers.dW1(1:Nindep,:);
        
    elseif ~isempty(rndNumbers) && isfield(rndNumbers.numbers,'Y')
        Y = rndNumbers.numbers.Y(1:Nindep,:);
        
        if any(~ismember(outVars,{'V','Y'}))
            error(['rBergomiClass:HybridScheme: When inputting ',...
                   'pre-simulated paths of the Y(t) process and not of ',...
                   'the Brownian increments dW_1(t) then you can only ',...
                   'request the output variables Y(t) and/or V(t).']);
        end
    else
        % Get covariance matrix for simulation of the Y(t) process:
        covM = rBergomiClass.CovMatrixHybrid(n,alpha,prec);
        
        % Factorize:
        try
            A = chol(covM, 'lower');
        catch
            % If matrix is not positive definite
            [V,D] = eig(covM);
            A = V*sqrt(D);
        end

        % Simulate (or extract) standard normals:
        if isempty(rndNumbers)
            Z = normrnd(0,1,2,nIntervals*Nindep);
        else
            % Readjust if necessary
            Z = rndNumbers.numbers.V_Gaussians(1:2,1:(nIntervals*Nindep));
        end

        % Simulate factors needed to construct Y(t):
        simFactors = A*Z;

        clear Z;

        Y1 = reshape(simFactors(2, :), Nindep, nIntervals);
        dW1 = reshape(simFactors(1, :), Nindep, nIntervals);

        clear simFactors;

        % Calculate weights for the convolution:
        nT_floored = idxRet(end)-1;
        ksUpper = (2:nT_floored);
        bk = ((ksUpper.^(alpha + 1) ...
             - (ksUpper - 1).^(alpha + 1))/(alpha + 1)).^(1/alpha);    
        weights = ConvertMatrix((bk/n).^(alpha),prec);

        % Calculate convolution
        if strcmpi(convMethod,'optimal')
           if nIntervals > 600
               convMethod = 'fft';
           else
               convMethod = 'conv2';
           end
        elseif ~strcmpi(convMethod,'conv2') && ~strcmpi(convMethod,'fft')
            error('rBergomiClass:HybridScheme: Invalid convolution method');
        end
        if strcmpi(convMethod,'conv2')
            Y2 = conv2(weights,dW1(:, 1:end-1), 'full');
        elseif strcmpi(convMethod,'fft')
            Y2 = ifft(fft([weights,zeros(1,nIntervals-1)])...
                    .*fft([dW1(1:Nindep,1:end-1),zeros(Nindep,nIntervals-1)],[],2),[],2);
        end
        clear weights;

        % Construct Y(t)
        Y = nan(Nindep,nIntervals,prec);
        Y(:, 1) = sqrt(2*H)*Y1(:, 1);
        Y(:, 2:end) = sqrt(2*H)*(Y1(:, 2:end) ...
                      + Y2(:, 1:nIntervals-1));
        clear Y1 Y2_with_extras;

    end
    
    % Store dW1
    if any(strcmpi('dW1',outVars))
        dW1Adj = SubsetMatrixNotStartingAtZero(dW1,idxRet,0,prec);
        if anti
            paths.dW1=[dW1Adj;-dW1Adj];
        else
            paths.dW1=dW1Adj;
        end
        if ~any(structfun(@isempty,paths));return;end
    end
    
    % Store Y(t)
    if any(strcmpi('Y',outVars))
        Yadj = SubsetMatrixNotStartingAtZero(Y,idxRet,0,prec);
        if anti
            paths.Y = [Yadj;-Yadj];
        else
            paths.Y = Yadj;
        end
        if ~any(structfun(@isempty,paths));return;end
    end    

    % Construct V(t)
    V = nan(N, nTimePts,prec);
    V(:, 1) = xi_eval(1);
    V(~idxAnti, 2:end) = bsxfun(@times,(xi_eval(2:end)'),...
                          exp( bsxfun(@plus,eta*Y, - 0.5*(eta^2)...
                          *(t(2:end)').^(1 - 2*gamm))));
    if anti
        V(idxAnti, 2:end) = bsxfun(@times,(xi_eval(2:end)'),...
                             exp( bsxfun(@plus,eta*(-Y),...
                             - 0.5*(eta^2)*(t(2:end)').^(1 - 2*gamm))));
    end

    clear Y;

    if any(strcmpi('V',outVars))
        paths.V = V(:,idxRet);
        if ~any(structfun(@isempty,paths));return;end            
    end

    % Construct the volatility process
    sig = sqrt(V);

    % Store the volatility process
    if any(strcmpi('sig',outVars))
        paths.sig = sig(:,idxRet);
        if ~any(structfun(@isempty,paths));return;end
    end

    % Compute the quadratic variation
    if any(strcmpi('QV',outVars))
        QV = dt*cumsum(V, 2);
        paths.QV = QV(:, idxRet);
        clear QV;
        if ~any(structfun(@isempty,paths));return;end
    end

    % Compute the stochastic integral int_0^t sqrt(V(s)) dW_1(s)
    if any(strcmpi('sig_dW1',outVars))
        int_sig_dW1 = NaN(N, nTimePts);
        int_sig_dW1(:, 1) = 0;
        if anti
            int_sig_dW1(:, 2:end) = cumsum(sig(:, 1:end-1) ...
                                                 .* [dW1;-dW1], 2);
        else
            int_sig_dW1(:, 2:end) = cumsum(sig(:, 1:end-1) ...
                                                 .* dW1, 2);
        end
        paths.int_sig_dW1 = int_sig_dW1(:, idxRet);  
        clear int_sig_dW1;
        if ~any(structfun(@isempty,paths));return;end
    end

    % Compute S_1(t)
    if any(strcmpi('S1',outVars)) || any(strcmpi('log_S1',outVars)) ...
            || any(strcmpi('S',outVars))
        dlog_S1 = nan(N, nIntervals,prec);
        dlog_S1(~idxAnti, :) = rho * sig(~idxAnti, 1:end-1)...
                    .* dW1 - 0.5 * (rho^2) * V(~idxAnti, 1:end-1) * dt;
        if anti
            dlog_S1(idxAnti, :) = rho * sig(idxAnti, 1:end-1)...
                    .* (-dW1) - 0.5 * (rho^2) * V(idxAnti, 1:end-1) * dt;
        end

        log_S1 = cumsum(dlog_S1, 2);

        % Store log(S_1(t))
        if any(strcmpi('log_S1',outVars))
         paths.log_S1 = SubsetMatrixNotStartingAtZero(log_S1,idxRet,0,prec);
        end
        clear dlog_S1 dW1;
        if ~any(structfun(@isempty,paths));return;end

        S1 = exp(log_S1);

        % Store S_1(t)
        if any(strcmpi('S1',outVars))
            paths.S1 = SubsetMatrixNotStartingAtZero(S1,idxRet,1,prec);
        end
        clear log_S1;
        if ~any(structfun(@isempty,paths));return;end
    end

    % Compute S(t)
    if any(strcmpi('S',outVars))
        % Simulate second brownian motion:
        if ~isempty(rndNumbers) && isfield(rndNumbers.numbers,'Wperp_Gaussians')
            dWperp = sqrt(dt).*rndNumbers.numbers.Wperp_Gaussians(1:Nindep,...
                                                                  1:nIntervals);
        else
            dWperp = sqrt(dt).*randn(Nindep,nIntervals,prec);
        end

        dlog_S_perp = NaN(N,nTimePts,prec);
        dlog_S_perp(~idxAnti, 1:end-1) = sqrt((1 - rho^2)) ...
                        * sig(~idxAnti, 1:end-1) .* dWperp ...
                        - 0.5 * (1 - rho^2) * V(~idxAnti, 1:end-1) * dt;
        if anti
            dlog_S_perp(idxAnti, 1:end-1) = sqrt((1 - rho^2)) ...
                        * sig(idxAnti, 1:end-1) .* (-dWperp) ...
                        - 0.5 * (1 - rho^2) * V(idxAnti, 1:end-1) * dt;
        end

        clear sig;

        S = nan(N,nTimePts,prec);
        S(:, 1) = 1;
        S(:, 2:end) = exp(cumsum(dlog_S_perp(:,1:end-1), 2)).*S1;
        clear V S1 dlog_S_perp;

        paths.S = S(:, idxRet);  
        if ~any(structfun(@isempty,paths));return;end            
    end

end

function A_sub = SubsetMatrixNotStartingAtZero(A,idxRet,val0,prec)
    A_sub = NaN(size(A,1),size(idxRet,1),prec);
    if idxRet(1)==1
        A_sub(:,2:end) = A(:,idxRet(2:end)-1);
        A_sub(:,1) = val0;
    else
        A_sub = A(:,idxRet-1);
    end
end





















