function [c,ceq] = rHestonNonNegReq(x,Txi)
% Description: Non-linear constraint function for calibration of the rough Heston model.
% It should be noted that x = (H, nu, rho, xi0,xi1, ..., xi27).

% Extract relevant parameters:
H = x(1);
v0 = x(4);
xi = x(5:end);

% Get theta curve:
theta = GetThetaFromXi(v0,H,Txi(2:end),xi);

% Check requirement:
[~, c_temp] = CheckNonNegReqTheta(v0,H,Txi(2:end),theta);
c = -c_temp;
ceq = 0;

end

