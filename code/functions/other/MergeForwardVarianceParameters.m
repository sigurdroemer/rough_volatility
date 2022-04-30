function idxGrp = MergeForwardVarianceParameters(Txi,Tobs)
% Description: Merges forward variance parameters depending on the
% expiries that are observed.
%
% Parameters:
%   Txi:  [Nx1 real] Forward variance maturities. Values must be 
%         strictly increasing.
%   Tobs: [Mx1 real] Observed expiries.
%
% Output:
%   idxGrp: [Nx1 integer] Indices that specify which values are 
%           merged together.
% 

if any(Txi == 0)
    idxGrp = [1;MergeForwardVarianceParameters(Txi(2:end),Tobs)];
    return;
end

% Construct grid:
t = [0;Txi];
n = size(t,1)-1;

% Merge time points:
idxGrp = NaN(size(Txi));
idxGrp(1) = 1;
for i=2:n
    if ~any(Tobs > t(i-1) & Tobs <= t(i)) || ~any(Tobs > t(i))
        idxGrp(i) = idxGrp(i-1);
    else
        idxGrp(i) = idxGrp(i-1) + 1;
    end
end

end