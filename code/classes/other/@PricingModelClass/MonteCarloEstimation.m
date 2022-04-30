function [p,idxCall,se] = MonteCarloEstimation(obj,k,ttm,cartProd,paths,...
                                              optType,condMC,cv,retSE,anti,prec)
% Description: Estimates prices of call and put options given a set of simulated 
% (risk-neutral) paths.
%
% Remark: The code is only valid if the asset price S(t) has dynamics of the 
% form
%
%   dS(t) = S(t)(r(t) - delta(t))dt + S(t)sqrt(V(t))dW_2(t)               (*)
%
% under the risk-neutral measure. Here r(t) and delta(t) are deterministic 
% functions (the risk-free interest rate and dividend yield resp.). Also, the 
% instantaneous variance V(t) must be adapted to the filtration generated some 
% Brownian motion W_1(t) and the Brownian motion W_2(t) must then be constructed 
% as 
%
%       W_2(t) = rho*W_1(t) + sqrt(1-rho^2)*W_perp(t)
%
% where W_perp(t) is another Brownian motion that is independent of W_1(t). The 
% parameter rho must lie in [-1,1] and must also be stored in the object
% property of the same name, i.e. in obj.rho.
%
% It is the user's responsibility to only use the code if these assumptions
% are fulfilled.
%
% Below the price estimation is explained in detail:
%  
% Note first that the code uses many of the variance reduction methods from
% (McCrickerd and Pakkanen, 2018) where the rough Bergomi model is considered. The 
% methods are however still valid in the more general situation explained above.
% 
% Let K be the strike of an option and let w = 1 if it is a call option and
% w = -1 if it is a put option. Let us also write/define
%
%   y(t) = (1 / t) * int_0^t r(s) ds
%   q(t) = (1 / t) * int_0^t delta(s) ds.
%
% Then under the stated assumptions we may rewrite the price as
%
%       exp(-y(T)*T)*E[(w*(S(T) - K))^+] 
%       = S(0)*exp(-q(T)*T)*E[(w*(Z(T) - exp(k)))^+]
%
% with k = log(K/F(T)) being the log-moneyness and F(T) being the expiry T 
% forward price and where Z(t) solves
%
%   dZ(t) = sqrt(V(t))Z(t)dW_2(t), Z(0) = 1.
% 
% It therefore suffices to perform the price estimation under the assumption
% of zero interest rate and dividend and an initial asset price of 1 and all 
% that using the strike exp(k) as explained above. The price can then easily be
% transformed to the more general setting by a simple multiplication. This is 
% also how the code is structured.
%
% In the following we therefore assume S(0) = 1 and r(t) = delta(t) = 0.
%
% Prices are then estimated as
% 
%   price est. = average( X_i + alpha*(Y_i-E[Y]); i=1,...,# paths)      (**)
% 
% where X is the main estimator of the option price, i.e. it will satisfy
% E[X] = E[{w*(S(T) - K)}^+]. See the parameter 'condMC' for the possible
% choices. Also, Y is a control variate (see the parameter 'cv' for the possible 
% choices). The 'alpha' parameter is chosen in an asymptotically optimal way, 
% see also (McCrickerd et al, 2018).
% 
% Parameters:
% k:        [Nx1 real] Log-moneyness.
%
% ttm:      [Mx1 real] Expiries.
%
% cartProd: [1x1 logical] If true then we return prices for the cartesian 
%           product of the k and ttm vectors. Else (and here assuming N = M)  
%           we return prices for each element (row-by-row).
%
% paths:    [1x1 struct] Struct containing some of the following fields (We let
%            A = # paths including antithetic ones, B = # time points).
%
%               o t:  [1xB real] Time points. MUST exactly include the unique 
%                     values in the 'ttm' parameter and be sorted in ascending
%                     order.
%
%               o S:  [AxB real] The stochastic process S(t) as defined in
%                     the description.
%
%               o S1: [AxB real] The stochastic process S_1(t) defined as
%
%                     S_1(t) = exp{rho*[int_0^t sqrt(V(u)) dW_1(u)] 
%                                  - 0.5*rho^2*[int_0^t V(u) du]}.
%
%               o QV: [AxB real] The stochastic process QV(t) defined as
%
%                     QV(t) = int_0^t V(u) du.
%
%           Paths MUST be simulated under the assumption of an initial value of 
%           1 and no drift, i.e. as in (*) except r(t)=delta(t)=0 and S(0)=1. 
%           The code will then automatically correct for this. If there are 
%           antithetic paths the first half of the paths must be the original 
%           paths and the second half the antithetic ones. The fields that
%           we require depend on the choice of the other parameters.
%           Specifically we require the fields 'S1' and 'QV' if either 
%           condMC = true or cv = 'timer_option'. We only require the field
%           'S' if condMC = false.
%
% optType:  [1x1 string] The type of option to estimate prices for. Possible  
%           values are 'call' (call options), 'put' (put options) and 'otm' 
%           (out-of-the-money options). Moneyness is here measured in terms
%           of forward-moneyness so an option is at-the-money when 
%           strike = forward price.
%
% condMC:   [1x1 logical] If false then X = {w*(S(T) - K)}^+ in (**), otherwise
%
%               X = E[(w*[S(T) - K])^+ | F_T^1 ] 
%                 = BS((1-rho^2)*int_0^T V(u) du;S_1(T),k).
%
%           where F_T^1 is the sigma algebra generated by the Brownian motion 
%           W_1(t) on the time interval [0,T] and BS(x;y,z) is the Black-Scholes 
%           price (put or call) with total variance x, current (spot) asset 
%           price y and strike z.
%
% cv:       [1x1 string] Control variate to be used. Options are
%
%           o 'timer_option': Here we set
%
%                   Y = BS(rho^2*[Q-QV(T)];S_1(T),k) 
%
%             in (**) where 
%
%                   Q = sup{ {QV(T)}_i , i=1,...,# paths}
%
%             is the supremum of the quadratic variation of S(t) over all
%             simulated paths.
%
%           o 'asset_price': Here we set Y = S_1(T) in (**) if condMC = true 
%             and otherwise set Y = S(T).
%
%           o 'none': Here we set Y = 0 in (**).
%
% retSE:    [1x1 logical] If set to true then we also compute standard errors.
%
% anti:     [1x1 logical] Set to true if half of the paths are antithetic, 
%           otherwise set it to false. In the former case the first half of the 
%           paths in the 'paths' parameter must be the original paths and the 
%           second half the antithetic ones. This is important for a correct 
%           computation of the standard errors.
%
% prec:     [1x1 string] Precision to be used in the computations. Options are 
%           'double' and 'single'.
%
% Output: 
%   p:       [NxM or Nx1 real] Option prices.
%   idxCall: [NxM or Nx1 logical] True for a call option, false for a put. 
%   se:      [NxM or Nx1 real] Standard errors.
%
% References:
%   o McCrickerd, R. and Pakkanen, M.S., Turbocharging Monte Carlo pricing for 
%     the rough Bergomi model. Quantitative Finance, 2018, 18(11), 1877-1886.
% 


    %% Initial computations:
    [p,idxCall,se] = deal([]);
    
    % Using cartProd = true with 1x1 dimensional inputs will collapse some of 
    % the matrix dimensions in the intermediate computations (which we do not
    % want). Thus this (valid) adjustment to the inputs:
    if size(k,1) == 1 && size(ttm,1) ~= 1
        k = repmat(k,size(ttm,1),1);
        cartProd = false;
    elseif size(ttm,1) == 1 && size(k,1) ~= 1
        ttm = repmat(ttm,size(k,1),1);
        cartProd = false;
    end
    
    rho = obj.rho;
    y = obj.y.DeepCopy();
    q = obj.q.DeepCopy();
    s0 = obj.s0;

    if strcmpi(cv,'timer_option')
        Qn = max(paths.QV, [], 1);
    end
    
    if isfield(paths, 'S1')
        numPaths = size(paths.S1, 1);
    else
        numPaths = size(paths.S, 1);
    end
    
    if cartProd
        %% Cartesian product:
        yields = ConvertMatrix(y.Eval(ttm),prec);
        divs = ConvertMatrix(q.Eval(ttm),prec);
        ZCBs = ConvertMatrix(exp(-yields.*ttm),prec);
        F = ConvertMatrix(s0.*exp((yields-divs).*ttm),prec);        
        
        k = ConvertMatrix(k,prec);
        k_grid = repmat(k, 1, size(ttm, 1));
        switch optType
            case 'call'
                idxCall = true(size(k_grid));
                idxCall_vec = true(size(k));
            case 'put'
                idxCall = false(size(k_grid));
                idxCall_vec = false(size(k));
            case 'otm'
                idxCall = k_grid>=0;
                idxCall_vec = k>=0;
        end

        numk = size(k,1);
        numT = size(ttm,1);
        if strcmpi(cv,'timer_option')
            totVar_Y = repmat((rho^2)*(Qn - paths.QV),1,1,1);
        end

        K = repmat(reshape(exp(k),1,1,numk),1,1);

        if strcmpi(cv,'timer_option')
            Y = nan(numPaths,numT,numk,prec);
            Y(:,:,idxCall_vec) =  bscall_3d(paths.S1,K(:,:,idxCall_vec),...
                                        0,1,totVar_Y);
            Y(:,:,~idxCall_vec) = bsput_3d(paths.S1,K(:,:,~idxCall_vec),...
                                        0,1,totVar_Y);
            clear totVar_Y;
        elseif strcmpi(cv,'asset_price')
            if condMC
                Y = paths.S1;
            else
                Y = paths.S;
            end
        end 
        
        X = nan(numPaths,numT,numk,prec);
        
        if condMC
            totVar_X = repmat((1 - rho^2)*paths.QV,1,1,1);
            if any(idxCall_vec)
                 X(:,:,idxCall_vec) = bscall_3d(paths.S1,K(:,:,idxCall_vec),...
                                                0,1,totVar_X);
            end
            if any(~idxCall_vec)
                 X(:,:,~idxCall_vec) = bsput_3d(paths.S1,K(:,:,~idxCall_vec),...
                                                0,1,totVar_X);
            end
            clear totVar_X;
        else
            if any(idxCall_vec)
                 X(:,:,idxCall_vec) = max(paths.S - K(:,:,idxCall_vec),0);
            end
            if any(~idxCall_vec)
                 X(:,:,~idxCall_vec) = max(K(:,:,~idxCall_vec) - paths.S,0);
            end
        end

        if strcmpi(cv,'timer_option')
            EY = nan(1,size(Qn,2),size(k,1),prec);
            EY(:,:,idxCall_vec) = bscall(1,squeeze(K(:,:,idxCall_vec))',...
                                       0,1,Qn'*rho^2);
            EY(:,:,~idxCall_vec) = bsput(1,squeeze(K(:,:,~idxCall_vec))',...
                                       0,1,Qn'*rho^2,0);
        elseif strcmpi(cv,'asset_price')
            EY = 1;
        end
        
        if anti
            % Average the original and antithetic paths:
            Nindep = size(X,1)/2;
            X = 0.5*(X(1:Nindep,:,:) + X(Nindep+1:end,:,:));
            if ~strcmpi(cv,'none')
                Y = 0.5*(Y(1:Nindep,:,:) + Y(Nindep+1:end,:,:));
            end
        end     
        
        mu_X = mean(X, 1);
        if ~strcmpi(cv,'none')
            mu_Y = mean(Y, 1);
            diff_Y = Y - mu_Y;
            alpha = -sum((X - mu_X).*diff_Y,1)./sum(diff_Y.^2,1);
            if any(any(isnan(alpha)))
                alpha(isnan(alpha)) = 0;
            end
            
            clear diff_Y;
            
        else
            alpha = 0;
            EY = 0;
            Y = 0;
            mu_Y = 0;
        end
        
        if retSE 
            % Compute standard error estimates of option prices
            Z = X + alpha.*(Y - EY);
            se = permute(squeeze(std(Z,1)),[2,1])./sqrt(size(Z,1));
            se = F'.*ZCBs'.*se;
        end
        
        clear X Y Z;

        % Estimate option prices
        if strcmpi(cv,'timer_option')
            p = (squeeze(mu_X) + squeeze(mu_Y).*squeeze(alpha)...
                  - squeeze(EY).*squeeze(alpha))';
        elseif strcmpi(cv,'asset_price')
            p = (squeeze(mu_X) + repmat(squeeze(mu_Y)', ...
                  1,size(alpha,3)).*squeeze(alpha) ...
                  - EY'.*squeeze(alpha))';
        elseif strcmpi(cv,'none')
            p = squeeze(mu_X)';
        end

        p = bsxfun(@times,F'.*ZCBs',p);
        
    elseif ~cartProd
        %% Non-cartesian product:
        k = ConvertMatrix(k,prec);
        K = exp(k);
        ttm = ConvertMatrix(ttm,prec);
        switch optType
            case 'call'
                idxCall = true(size(k));
            case 'put'
                idxCall = false(size(k));
            case 'otm'
                idxCall = k>=0;
        end
        uniqT = sort(unique(ttm));
        p = NaN(size(k,1),1);
        if retSE
            se = NaN(size(p));
        end
        for i=1:size(uniqT,1)
            idxT = ttm == uniqT(i);
            numStrikes = sum(idxT);
            if condMC || strcmpi(cv,'timer_option')
                s1 = paths.S1(:, i);
            end
            if ~condMC
                s = paths.S(:,i);
            end
                
            idxCallSub = idxCall(idxT);
            Ksub = K(idxT);

            if strcmpi(cv,'timer_option')
                totVar_Y = (rho^2)*( Qn(i) - paths.QV(:,i));
                Y = nan(numPaths, numStrikes, prec);
                if any(idxCallSub)
                    Y(:,idxCallSub) = bscall(s1,Ksub(idxCallSub)',0,1,...
                                             totVar_Y);
                end
                if any(~idxCallSub)
                    Y(:,~idxCallSub) = bsput(s1,Ksub(~idxCallSub)',0,1,...
                                             totVar_Y,0);
                end
                clear totVar_Y;
            elseif strcmpi(cv,'asset_price')
                if condMC
                    Y = s1;
                else
                    Y = s;
                end
            elseif strcmpi(cv,'none')
                Y = 0;
                EY = 0;
                alpha = 0;
                mu_Y = 0;
            end
            
            X = nan(numPaths, numStrikes, prec);
            if condMC == true
                totVar_X = (1 - rho^2)*paths.QV(:,i);
                if any(idxCallSub)
                    X(:,idxCallSub) = bscall(s1,Ksub(idxCallSub)',0,1,...
                                             totVar_X);
                end
                if any(~idxCallSub)
                    X(:,~idxCallSub) = bsput(s1,Ksub(~idxCallSub)',0,1,...
                                             totVar_X,0);
                end
            else
                if any(idxCallSub)
                    X(:,idxCallSub) = max(s-Ksub(idxCallSub)',0);
                end
                if any(~idxCallSub)
                    X(:,~idxCallSub) = max(Ksub(~idxCallSub)'-s,0);
                end
            end

            if strcmpi(cv,'timer_option')
                EY = nan(1, numStrikes, prec);
                if any(idxCallSub)
                    EY(idxCallSub) = bscall(1, Ksub(idxCallSub)',...
                                            0, 1, Qn(i)*rho^2)';
                end
                if any(~idxCallSub)
                    EY(~idxCallSub) =  bsput(1,Ksub(~idxCallSub)',...
                                             0, 1, Qn(i)*rho^2);
                end
            elseif strcmpi(cv,'asset_price')
                EY = 1;
            end
            
            if anti
                % Average the original and antithetic paths:
                Nindep = size(X,1)/2;
                X = 0.5*(X(1:Nindep,:) + X(Nindep+1:end,:));
                if ~strcmpi(cv,'none')
                    Y = 0.5*(Y(1:Nindep,:) + Y(Nindep+1:end,:));
                end
            end
            mu_X = mean(X, 1);
            if ~strcmpi(cv,'none')
                mu_Y = mean(Y, 1);
                diff_Y = Y - mu_Y;
                alpha = -sum((X - mu_X).*diff_Y,1) ./ sum(diff_Y.^2,1);
                if any(isnan(alpha))
                   alpha(isnan(alpha)) = 0;
                end
            end
            
            % Adjust by dividend yield and interest rate
            yields = ConvertMatrix(y.Eval(uniqT(i)),prec);
            divs = ConvertMatrix(q.Eval(uniqT(i)),prec);
            ZCB = ConvertMatrix(exp(-yields.*uniqT(i)),prec);
            F = ConvertMatrix(s0.*exp((yields - divs).*uniqT(i)),prec);
            
            p(idxT) = ZCB*F*(mu_X + mu_Y.*alpha - EY.*alpha)';
            
            % Standard Errors:
            if retSE
               % Compute standard error estimates of option prices
               Z = X + bsxfun(@times,alpha,(Y - EY));
               se(idxT) = permute(squeeze(std(Z,1)),[2,1])./sqrt(size(Z,1));
               se(idxT) = bsxfun(@times,F'.*ZCB',se(idxT));
            end
            
        end

    end
    
end 

















