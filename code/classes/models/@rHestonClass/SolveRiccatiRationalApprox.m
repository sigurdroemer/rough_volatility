function Dalphah = SolveRiccatiRationalApprox(~,a,t,H,nu,rho)
% Description: Returns the solution to the fractional Riccati equation
%              
%    D^(alpha)h(a,t) = -0.5*a*(a + i) + i*rho*nu*a*h(a,t) + 0.5*nu^2*h^2(a,t)
%
% subject to the initial condition
%
%    I^(1-alpha)h(a,0) = 0
%
% and all that with alpha = H + 0.5. See also equation (1.3) in (Gatheral
% and Radoicic, 2019).
%
% Parameters:
%   a:   [Nx1 real or complex] See the description.
%   t:   [Mx1 real] See the description.
%   H:   [1x1 real] See the description.
%   nu:  [1x1 real] See the description.
%   rho: [1x1 real] See the description.
%
% Output: 
%   Dalphah: [NxM real or complex] The value of D^(alpha)h(a,t).
%
% References: 
%   o Gatheral, J. and Radoicic, R., Rational Approximation of the rough 
%     Heston solution. 2019, International Journal of Theoretical and Applied 
%     Finance, 22(3), 1950010.
    
    alpha = 0.5 + H;

    %% b coefficients
    coeff = @(j,a)( ((gamma(1+j*alpha))/(gamma(1+(j+1)*alpha)))*beta(j,a,alpha,rho)  ); 

    b1 = coeff(0,a).';
    b2 = coeff(1,a).';
    b3 = coeff(2,a).';

    %% q coefficients:
    A = sqrt( a.*(a+1i) - rho^2*a.^2 ).';
    rm = -1i*rho*a.'-A;
    rp = -1i*rho*a.'+A;

    coeff = @(k,a)(rm.*gamm(k,a.',alpha,rho)./(A.^k*gamma(1 - k*alpha)));
    g0 = coeff(0,a);
    g1 = coeff(1,a);
    g2 = coeff(2,a);

    %% Coefficients from rational approximation:
    p1 = b1;

    nom = b1.^3.*g1 + b1.^2.*g0.^2 + b1.*b2.*g0.*g1 - b1.*b3.*g0.*g2 ...
                    + b1.*b3.*g1.^2 + b2.^2.*g0.*g2 - b2.^2.*g1.^2 + b2.*g0.^3;
    denom = b1.^2.*g2 + 2.*b1.*g0.*g1 + b2.*g0.*g2 - b2.*g1.^2 + g0.^3;
    p2 = nom./denom;

    nom = b1.^2.*g1 - b1.*b2.*g2 + b1.*g0.^2 - b2.*g0.*g1 - b3.*g0.*g2 ...
                    + b3.*g1.^2;
    denom = b1.^2.*g2 + 2.*b1.*g0.*g1 + b2.*g0.*g2 - b2.*g1.^2 + g0.^3;
    q1 = nom./denom;

    nom = b1.^2.*g0 - b1.*b2.*g1 - b1.*b3.*g2 + b2.^2.*g2 + b2.*g0.^2 ...
                    - b3.*g0.*g1;
    denom = b1.^2.*g2 + 2.*b1.*g0.*g1 + b2.*g0.*g2 - b2.*g1.^2 + g0.^3;
    q2 = nom./denom;

    nom = b1.^3 + 2.*b1.*b2.*g0 + b1.*b3.*g1 - b2.^2.*g1 + b3.*g0.^2;
    denom = b1.^2.*g2 + 2.*b1.*g0.*g1 + b2.*g0.*g2 - b2.*g1.^2 + g0.^3;
    q3 = nom./denom;

    p3 = g0.*q3;

    %% Compute solution:
    h33_ay = @(y)((y*p1 + y.^2*p2 + y.^3*p3) ./ (1 + y*q1 + y.^2*q2 + y.^3*q3));
    h33_ax = @(x)( h33_ay(x.^(alpha)) );
    Da_h33_ax = @(x)( 0.5*( h33_ax(x) - rm).*( h33_ax(x) - rp)  );    
    Dalphah = Da_h33_ax(t*(nu^(1/alpha))).';

end

function beta_k = beta(k,a,alpha,rho)
    % First term:
    if k==0
        beta_k = -0.5*a.*(a+1i);
        return;
    end
    
    % Compute recursively:
    beta_k = complex(zeros(size(a)),0);
    for i=0:k-2
        for j=0:k-2
            if i+j==k-2
                fac1 = gamma(1+i*alpha)/gamma(1 + (i+1)*alpha);
                fac2 = gamma(1+j*alpha)/gamma(1+(j+1)*alpha);
                beta_k = beta_k + bsxfun(@times,beta(i,a,alpha,rho),...
                                         beta(j,a,alpha,rho))*fac1*fac2;
            end
        end
    end
    beta_k = 0.5*beta_k;
    beta_k = beta_k + 1i*rho*a.*((gamma(1+(k-1)*alpha))/(gamma(1+k*alpha)))...
                                .*beta(k-1,a,alpha,rho);

end

function gamma_k = gamm(k,a,alpha,rho)
    % First two terms:
    if k==0
        gamma_k = complex(ones(size(a)),0);
        return;
    elseif k==1
        gamma_k = -1*complex(ones(size(a)),0);
        return;
    end
    
    A = sqrt( a.*(a+1i) - rho^2*a.^2 );
    rm = -1i*rho*a-A;
    
    % Sum terms recursively:
    gamma_k = complex(zeros(size(a)),0);
    for i=1:k
        for j=1:k
            if i+j==k
                fac = gamma(1-k*alpha)/(gamma(1-i*alpha)*gamma(1-j*alpha));
                gamma_k = gamma_k + bsxfun(@times,gamm(i,a,alpha,rho),...
                                           gamm(j,a,alpha,rho))*fac;
            end
        end
    end
    
    gamma_k =  - gamm(k-1,a,alpha,rho) + (rm./(2*A)).*gamma_k;
    
end