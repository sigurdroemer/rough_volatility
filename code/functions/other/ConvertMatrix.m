function x_conv = ConvertMatrix(x,prec)
% Description: Converts a numeric matrix according to the specified precision. 
% I.e. if the precision is 'single' (resp. 'double') we convert the input 
% matrix to single (resp. double) precision.
%
% Parameters:
%      x: [NxM real] Numeric matrix to be converted.
%   prec: [1x1 string (optional)] Options are 'double' (default) and 'single'.
%
% Output: 
%    x_conv: [NxM real] Numeric matrix converted.
%

   if ~exist('prec','var')||isempty(prec);prec='double';end

   if strcmpi(prec, 'single')
       x_conv = single(x);
   elseif strcmpi(prec, 'double')
       x_conv = double(x);
   end                             

end