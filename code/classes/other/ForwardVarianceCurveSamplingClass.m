classdef ForwardVarianceCurveSamplingClass < GenericSuperClass
% Description: Implements functionality to sample forward variance 
% curves.
    
methods
    function obj = ForwardVarianceCurveSamplingClass(varargin)
    % Description: Constructor.
        obj.ParseConstructorInputs(varargin{:});
    end
    function [xi,vs_rate,curves] = SampleCurves(obj,nSamples,Ts,type,perc,vol0_low,...
                                                vol0_high,vs_rate_low,vs_rate_high,...
                                                kappa_low,kappa_high,eps_noise,xi_bounds,...
                                                num_flat_sections,retCurves)
    % Description: Samples forward variance curves.
    %
    % Parameters:
    %   nSamples:     [1x1 integer] The number of samples to produce.
    %   Ts:           [Nx1 real] Grid points between which we assume the
    %                 forward variance curve flat. Sorted in ascending
    %                 order.
    %   type:         [Mx1 cell] Each element is a string specifying which
    %                 method to simulate from. Options are: 'Heston','flat'
    %                 'linear','independent','piecewise_flat'.
    %   perc:         [Mx1 real] The probability a sample will come from
    %                 each group.
    %   vol0_low:     [Mx1 real] Lower bounds for the initial volatility.
    %   vol0_high:    [Mx1 real] Upper bounds for the initial volatility.
    %   vs_rate_low:  [Mx1 real] Lower bounds for the final variance swap rate.    
    %   vs_rate_high: [Mx1 real] Upper bounds for the final variance swap rate.
    %   kappa_low:    [Mx1 real] Lower bounds for kappa. Only used for the
    %                 method 'Heston'.
    %   kappa_high:   [Mx1 real] Upper bounds for kappa. Only used for the
    %                 method 'Heston'.
    %   eps_noise:    [Mx1 real] Sizes of the noise.
    %   xi_bounds:    [1x2 real] Bounds that forward variance values should
    %                 be restricted to (even after adding noise).
    %   num_flat_sections:
    %                 [Mx1 integer] Only used by the method 'piecewise_flat'.
    %                 Specifies the number of flat sections.
    %   retCurves:    [1x1 logical (optional)] If true we also return a 
    %                 nSamples x 1 cell array of CurveClass objects for each of 
    %                 the forward variance curves. Default is false.
    %
    % Output:
    %   xi:      [nSamples x N real] Forward variance curves.
    %   vs_rate: [nSamples x N real] Variance swap rates.
    %    
    
        if ~exist('retCurves','var');retCurves = false;end
        M = size(perc,1);
        
        if abs(sum(perc) - 1) > eps
            error(['ForwardVarianceCurveSamplingClass:SampleCurves:',...
                   ' Percentages must sum to 1.']);
        end
        
        % Allocate the samples:
        U = unifrnd(0,1,nSamples,1);
        num_per_grp = NaN(M,1);
        lb = 0;
        for i=1:size(perc,1)-1
            num_per_grp(i) = sum(lb <= U & U < lb + perc(i));
            lb = lb + perc(i);
        end
        num_per_grp(end) = nSamples - sum(num_per_grp(1:end-1));
    
        % Sample curves group-by-group:
        [xi,vs_rate,curves] = deal([]);
        for i=1:size(perc,1)
            if num_per_grp(i) == 0
                continue;
            end
            switch type{i}
                case 'Heston'
                    vol0_rng = [vol0_low(i),vol0_high(i)];
                    vs_rate_rng = [vs_rate_low(i),vs_rate_high(i)];
                    kappa_rng = [kappa_low(i),kappa_high(i)];
                    xi_tmp = obj.SampleHestonCurves(...
                                                    num_per_grp(i),...
                                                    kappa_rng,vol0_rng,Ts,...
                                                    vs_rate_rng);
                case 'flat'
                    lb = vol0_low(i);
                    ub = vol0_high(i);
                    if lb ~= vs_rate_low(i) || ub ~= vs_rate_high(i)
                        error(['ForwardVarianceCurveSamplingClass:SampleCurves:',...
                               ' When using type = ''flat'' the lower, respectively,',...
                               ' upper bounds for the initial volatility and ',...
                               ' final variance swap rate must be the same.']);
                    end
                    xi_tmp = repmat(unifrnd(lb,ub,num_per_grp(i),1).^2,1,size(Ts,1));                
                    
                case 'linear'
                    % Sample end-points:
                    xi0 = unifrnd(vol0_low(i),vol0_high(i),num_per_grp(i),1).^2;
                    xiEnd = unifrnd(vs_rate_low(i),vs_rate_high(i),num_per_grp(i),1).^2;
                    
                    % Interpolate:
                    xi_tmp = NaN(num_per_grp(i),size(Ts,1));
                    for j=1:size(xi0,1)
                        xi_tmp(j,:) = interp1([0;Ts(end)],[xi0(j);xiEnd(j)],Ts)';
                    end
                    
                case 'piecewise_flat'
                    lb = vol0_low(i);
                    ub = vol0_high(i);
                    if lb ~= vs_rate_low(i) || ub ~= vs_rate_high(i)
                        error(['ForwardVarianceCurveSamplingClass:SampleCurves:',...
                               ' When using type = ''piecewise_flat'' the lower, ',...
                               ' respectively, upper bounds for the initial volatility',...
                               ' and final variance swap rate must be the same.']);
                    end                    
                    xi_tmp = NaN(num_per_grp(i),size(Ts,1));
                    n = num_flat_sections(i)-1;
                    for j=1:size(xi_tmp,1)
                        % Sample separating points:
                        U = sort(unifrnd(Ts(1),Ts(end),1,n));
                        idx = NaN(size(U));
                        for l=1:size(U,2)
                            [~,idx(l)] = min(abs(Ts - U(l)));
                        end
                        idx_ext = unique([1,idx,size(Ts,1)]);
                        
                        % Sample values:
                        base_val = unifrnd(lb,ub,1,size(idx_ext,2)-1).^2;
                        for l=1:size(idx_ext,2)-1
                            xi_tmp(j,idx_ext(l):idx_ext(l+1)) = base_val(l);
                        end
                    end
                    
                case 'independent'
                    lb = vol0_low(i);
                    ub = vol0_high(i);
                    if lb ~= vs_rate_low(i) || ub ~= vs_rate_high(i)
                        error(['ForwardVarianceCurveSamplingClass:SampleCurves:',...
                               ' When using type = ''independent'' the lower, respectively,',...
                               ' upper bounds for the initial volatility and ',...
                               ' final variance swap rate must be the same.']);
                    end
                    xi_tmp = unifrnd(lb,ub,num_per_grp(i),size(Ts,1)).^2;
                    
                otherwise 
                    error(['ForwardVarianceCurveSamplingClass:SampleCurves:',...
                           ' Sampling type is not supported.']);
            end
            
            % Add noise and apply xi_bounds
            err = unifrnd(-eps_noise(i),eps_noise(i),size(xi_tmp,1),size(Ts,1));
            xi_tmp = (sqrt(xi_tmp) + err).^2;

            % Truncate if needed:
            xi_tmp(xi_tmp < xi_bounds(1)) = xi_bounds(1);
            xi_tmp(xi_tmp > xi_bounds(2)) = xi_bounds(2);                    

            % Compute vs rate
            dt = [Ts(1);diff(Ts)];
            vs_rate_tmp = sqrt(cumsum(xi_tmp.*dt.',2)./Ts.');

            % Construct curves if needed:
            if retCurves
                curves_tmp = cell(size(xi_tmp,1),1);
                for j=1:size(xi_tmp,1)
                    curves_tmp{j} = CurveClass('gridpoints',Ts,...
                                           'values',xi_tmp(j,:)');
                end
            else
               curves_tmp = [];
            end
            
            % Store samples:
            vs_rate = [vs_rate;vs_rate_tmp];
            xi = [xi;xi_tmp];
            curves = [curves;curves_tmp];
        end
        
        % Shuffle the samples:
        idx = randperm(size(xi,1));
        xi = xi(idx,:);
        vs_rate = vs_rate(idx,:);
        if retCurves;curves = curves(idx);end
    
    end    
    function xi = SampleHestonCurves(obj,N,kappa_rng,vol0_rng,...
                                              Ts,vs_rate_rng)
    % Description: Samples forward variance curves using the structure of forward 
    % variances under Heston.
    %
    % Parameters:
    %   N:           [1x1 integer] Number of curves to sample.
    %   kappa_rng:   [1x2 real] The kappa parameter will be simulated uniformly 
    %                from this range.
    %   vol0_rng:    [1x2 real] The initial volatility will be simulated 
    %                uniformly from this range.
    %   Ts:          [Mx1 real] Grid points.
    %   vs_rate_rng: [1x2 real] The final value (at the expiry T) of the 
    %                variance swap quotes will be simulated from this range 
    %                (in volatility terms) using a uniform distribution. 
    %
    % Output:
    %   xi:      [N x M real] Forward variance curves.
    %

        % Sample Heston parameters:
        [v0, theta, kappa] = obj.SampleHestonParameters(N,kappa_rng,...
                                                       vol0_rng,Ts(end),...
                                                       vs_rate_rng);


        % Compute values at grid points:
        vs = obj.GetHestonVarianceSwapCurves(theta,v0,kappa,Ts);
        dt = [Ts(1);diff(Ts)];
        xi = [vs(:,1),diff(vs,1,2)]./ dt';

    end
    function [v0,theta,kappa] = SampleHestonParameters(obj,N,kappa_rng,...
                                                         vol0_rng,T,...
                                                         vs_rate_rng)
    % Description: Samples Heston parameters.
    %
    % Parameters:
    %   N:           [1x1 integer] Number of curves to sample.
    %   kappa_rng:   [1x2 real] The kappa parameter will be simulated uniformly 
    %                from this range.
    %   vol0_rng:    [1x2 real] The initial volatility will be simulated 
    %                uniformly from this range.
    %   T:           [1x1 real] Expiry at which to (attempt to) force the 
    %                variance swap curve through a specific value.
    %   vs_rate_rng: [1x2 real] The final value (at the expiry T) of the 
    %                variance swap quotes will be simulated from this range 
    %                (in volatility terms) using a uniform distribution. 
    %
    % Output:
    %   v0:    [Nx1 real] Initial variances.
    %   theta: [Nx1 real] Long term variances.
    %   kappa: [Nx1 real] Mean reversion speeds.
    %

        Nmax = 100;

        [v0,theta,kappa] = deal(NaN(N,1));
        for i=1:N
               valid = false;
               while ~valid
                    % Simulate short end:
                    v0_tmp = unifrnd(vol0_rng(1),vol0_rng(2)).^2;

                    % Simulate the long term rate: 
                    vs_rate_tmp = unifrnd(vs_rate_rng(1),vs_rate_rng(2));

                    j = 0;
                    while ~valid && j < Nmax
                        % Simulate kappa
                        kappa_tmp = unifrnd(kappa_rng(1),kappa_rng(2),1,1);

                        % Invert model:
                        [theta_tmp,valid] = obj.InvertModel(kappa_tmp,...
                                                v0_tmp,T.*vs_rate_tmp.^2,T);

                        j = j + 1;
                    end

                end
                v0(i) = v0_tmp;
                theta(i) = theta_tmp;
                kappa(i) = kappa_tmp;
        end
    end
    function [theta,valid] = InvertModel(~,kappa,v0,vs,T)
    % Description: Finds the theta parameter s.t. the variance swap quotes
    % (under Heston) goes through the point (T,VS(T)) where VS(T) denotes
    % the variance swap quote and T > 0 is some chosen future expiry. The
    % other parameters kappa and v0 are fixed before hand.
    %
    % Remark: There will not always be a solution to this problem. If a
    % valid solution is found this will be indicated by the output 'valid'
    % being set to true.
    %
    % Parameters:
    %   kappa: [1x1 real] Mean reversion speed.
    %   v0:    [1x1 real] Initial variance.
    %   vs:    [1x1 real] Variance swap quote.
    %   T:     [1x1 real] Expiry of variance swap quote.
    %
    % Output: 
    %   theta: [1x1 real] Long term variance.
    %   valid: [1x1 logical] True if inverted parameters are valid, otherwise false.
    %
    
        f =  (1 - exp(-kappa*T))./(kappa) ;

        dummy1 = vs  - v0.*f;
        dummy2 = T - f;

        theta = dummy1 ./ dummy2;

        valid = ~isnan(theta) && isfinite(theta) && (theta > 0);

    end
    function vs = GetHestonVarianceSwapCurves(obj,theta,v0,kappa,T)
    % Description: Returns variance swap quotes under the Heston model and
    % that for multiple input parameters.
    %
    % Parameters:
    %   theta: [Mx1 real] Long term variance.
    %   v0:    [Mx1 real] Initial variance level.
    %   kappa: [Mx1 real] Mean reversion speed.
    %   T:     [Nx1 real] Future dates.
    %
    % Output:
    %   vs: [MxN real] Variance swap quotes.
    %    

        vs = NaN(size(theta,1),size(T,1));
        for i=1:size(theta,1)
            vs(i,:) = obj.GetHestonVarianceSwaps(theta(i),v0(i),kappa(i),T).';
        end

    end        
    function vs = GetHestonVarianceSwaps(~,theta,v0,kappa,T)
    % Description: Returns variance swap quotes under the Heston model.
    %
    % Parameters:
    %   theta: [1x1 real] Long term variance.
    %   v0:    [1x1 real] Initial variance level.
    %   kappa: [1x1 real] Mean reversion speed.
    %   T:     [Nx1 real] Future dates.
    %
    % Output:
    %   vs: [Nx1 real] Variance swap quotes.
    %

        vs = T.*theta + (v0 - theta).*((1 - exp(-kappa.*T))./kappa);

    end
end

end

