function [out, idxNotValid] = ConvertTrainingDataToIV(data,header,s0,k_vec,...
                                                      T_vec,inclSE)
% Description: A simple function to convert a training data set with call
% option prices (and possibly standard errors) to implied volatilities.
% Standard errors (if relevant) are also converted to implied volatility.
% Assumes zero interest rate and dividend.
%
% Parameters:
%   data:   [NxM real] Matrix containing input parameters, call option
%           prices, and standard errors for N samples.
%   header: [1xM cell] The header for data (a cell array of strings).
%   s0:     [1x1 real] Spot price.
%   k_vec:  [Lx1 real] Log moneyness value.
%   T_vec:  [Lx1 real] Expiries.
%   inclSE: [1x1 logical] True if data contains standard errors.
%

    % Find when actual values start:
    idFirstVal = 0;
    for i=1:size(header,2)
        hd = header{i};
        if length(hd) >= 6 && strcmpi(hd(1:6),'price_')
            idFirstVal = i;
            break;
        end
    end

    nVals = size(k_vec,1);
    vals = data(:,idFirstVal:idFirstVal+nVals-1);
    inputs = data(:,1:idFirstVal-1);
    valsIV = NaN(size(vals));
    
    % Compute implied volatilities:
    K = s0.*exp(k_vec);
    for i=1:size(data,1)
        disp(['Iteration ', num2str(i), ' out of ', num2str(size(data,1))]);
        iv = blsimpv(s0,K,0,T_vec,vals(i,:)','Limit',5,'Yield',0,'Class',1);
        idxZeroNaN = iv==0 | isnan(iv);
        if any(idxZeroNaN)
            iv(idxZeroNaN) = blsimpv(s0,K(idxZeroNaN),0,T_vec(idxZeroNaN),vals(i,idxZeroNaN)',...
                            'Limit',5,'Yield',0,'Class',1,'Method','Search','Tolerance',10^(-20));
        end
        valsIV(i,:) = iv';        
    end
    
    % Check if all is valid:
    idxNotValid = any(isnan(valsIV),2) | any(valsIV <= 0,2);    
    
    % Compute iv standard errors if needed
    if inclSE
        se = data(:,idFirstVal + nVals:end);
        iv_se = NaN(size(valsIV));
        for i=1:size(data,1)
            vegas = BSGreek('vega',[],s0,K,0,T_vec,valsIV(i,:)',0);
            iv_se(i,:) = se(i,:)./vegas';
        end
        out = [inputs,valsIV,iv_se]; 
    else
        out = [inputs,valsIV]; 
    end
    

end

