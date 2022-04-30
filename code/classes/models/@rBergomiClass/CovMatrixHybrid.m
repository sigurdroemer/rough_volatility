function covM = CovMatrixHybrid(n,alpha,prec)
% Description: Returns the covariance matrix needed for the hybrid scheme
% of (Bennedsen et al., 2017) with their kappa = 1.
%
% Parameters:
%   n:      [1x1 integer] Number of steps per year.
%   alpha:  [1x1 real] Roughness index, equals Hurst exponent minus 1/2.
%   prec:   [1x1 string] Precision, options are 'double' and 'single'.
%
% Output: 
%   covM:   [2x2 real] Covariance matrix.
%
% References: 
%   o Bennedsen, M., Lunde, A. and Pakkanen, M.S., Hybrid scheme for Brownian 
%     semistationary procesess. Finance and Stochastics, 2017, 21(4), 931-965.
% 

    covM = nan(2,2,prec);
    covM(1, 1) = 1 / n;
    for j=2:2
       covM(1, j) = ( (j-1)^(alpha + 1) - (j-2)^(alpha + 1) ) ...
                    / ((alpha + 1)*n^(alpha + 1));
       covM(j ,j) = ( (j-1)^(2*alpha + 1) - (j-2)^(2*alpha + 1) )  ...
                    / ((2*alpha + 1)*n^(2*alpha + 1));
    end

    covM(isnan(covM))=0;
    covMDiagZero =  covM - diag(diag(covM));
    covM = covM + triu(covMDiagZero)';
    
end
