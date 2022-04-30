%% Clear workspace and load code:
clear;
project_folder = fileparts(fileparts(fileparts(matlab.desktop.editor.getActiveFilename)));
addpath(genpath(project_folder));

%% Define an extended rough Bergomi model (and inspect the object)
% Remark: We start by covering the simplest way to get started. More
% advanced features are covered further down in the script. It is recommended 
% to run the code one line at the time to get a better feel for how to work 
% with the model.

% Define a model:
model = rBergomiExtClass('alpha',-0.45,'beta',-0.35,'xi',0.15^2,'eta',2.3,'rho',-0.8,'s0',100);

model

% Curves are stored as CurveClass objects:
model.y
model.q
model.xi

% Note how the yield curve and dividend yield by default is assumed flat
% at zero. The forward variance curve is set to the fixed value previously
% inputted. 

% To start we want to use a low number of paths when running the Monte Carlo
% algorithm. This is to ensure that you have enough RAM for the example 
% computations. You can of course change this if you wish.
model.pricerSettings.N = 25000;

%% Compute implied volatility:
% Here we illustrate how to compute implied volatilities.

% Log-moneyness values, i.e. log(strike/forward):
k = (-0.3:0.05:0.1)'; 

% Expiries:
T = [0.05;0.2;0.5;1;2;3]; 

% Return outputs for the cartesian product of the inputs:
cartProd = true;

% Compute volatilities:
ivSurf = model.GetPrices(k,T,cartProd);
ivSurf

% Plot:
figure;
for i=1:size(ivSurf.k,2)
    plot(ivSurf.k(:,i),ivSurf.iv(:,i),'-o','DisplayName',...
         ['T = ',num2str(T(i))],'LineWidth',1.25);
    hold on;
end
hold off;
title('Extended rough Bergomi smiles');
xlabel('Log-moneyness');
ylabel('Implied volatility');
hleg = legend();
title(hleg,'Expiry');

%% Compute option prices:
% Here we illustrate how one can also compute e.g. put option prices
% directly.

k = (-0.3:0.01:0.2)';
T = (0.2:0.2:3)';
[p,k_mat,T_mat] = model.GetPrices(k,T,true,'priceType','price',...
                                           'optionType','put');

p

% Also, if you want to use strikes instead of log-moneyness you need to compute
% them yourself like this:
F = model.s0.*exp((model.y.Eval(T) - model.q.Eval(T)).*T);
K_mat = exp(k_mat).*F.';

% Now we plot the result:
figure;
surf(K_mat,T_mat,p);
title('Put option prices under extended rough Bergomi');
xlabel('Strike');
ylabel('Expiration');
zlabel('Price');

%% A quick demo of some other functionality:
% Here we illustrate the various types of outputs that can be produced:
k = (-0.3:0.05:0.2)';
T = [0.5;0.75;1;2;3];
p = model.GetPrices(k,T,true,'priceType','price','optionType','put')
p = model.GetPrices(k,T,true,'priceType','price','optionType','call')
p = model.GetPrices(k,T,true,'priceType','implied_volatility')
p = model.GetPrices(k,T,true,'priceType','implied_volatility_surface')

% You can also request only out-of-the-money options (moneyness is here
% defined relative to the forward price):
[p,~,~,idxCall] = model.GetPrices(k,T,true,'priceType','price','optionType','otm');
p

% The true/false matrix below tells you which prices are for call options:
idxCall

% The input vectors can also be interpreted row-by-row like this:
k = [0;-0.2;0;0.1;0.15];
T = [0.1;0.1;0.2;0.2;0.5];
cartProd = false;
p = model.GetPrices(k,T,cartProd,'priceType','implied_volatility');
p

% You can also get standard errors. They will automatically be expressed in
% volatility terms if you request implied volatilities or in prices (puts or 
% calls) if you request that.

% Let us set a small number of paths for better illustration:
model.pricerSettings.N = 10000;

% Compute prices and standard errors:
k = (-0.3:0.01:0.2)';
T = 0.5;
[p,~,~,~,se] = model.GetPrices(k,T,true,'priceType','price',...
                                              'optionType','otm',...
                                              'standardError',true);

% Plot the results:
figure;
plot(k,[p-se,p,p+se]);
xlabel('Log-moneyness');
ylabel('Price');
title(['Prices of out-of-the-money options (T = ',num2str(T),')']);
legend('Price - s.e.', 'price','price + s.e.');

% We can do the same for the implied volatility:
[iv,~,~,~,se] = model.GetPrices(k,T,true,'priceType','implied_volatility',...
                                              'standardError',true);

% Plot the results:
figure;
plot(k,[iv-se,iv,iv+se]);
xlabel('Log-moneyness');
ylabel('Implied volatility');
title(['Implied volatility (T = ',num2str(T),')']);
legend('Implied vol. - s.e.', 'Implied vol.','Implied vol. + s.e.');

% Reset # paths:
model.pricerSettings.N = 25000;
                                          
%% Using more general curve objects:
% To define a more general forward variance curve we do as follows:
t_pts = [0.1;0.25;0.75;1;2];
xi_vals = [0.1;0.15;0.25;0.3;0.31].^2;
xi_curve = CurveClass('gridpoints',t_pts,'values',xi_vals);

% The curve looks like this:
t_eval = (0:0.01:3)';
xi_eval = xi_curve.Eval(t_eval);
figure;
plot(t_eval,xi_eval);
xlabel('Expiry');ylabel('Forward variance');title('Forward variance curve');
ylim([0,max(xi_eval)*1.1]);

% This can then be set in the object like this:
model.xi = xi_curve;

% Or we can simply define a new model as:
model = rBergomiExtClass('alpha',-0.45,'beta',-0.35,'xi',xi_curve,'eta',2.3,'rho',-0.8,'s0',100);
model.pricerSettings.N = 10000;

% The same approach goes for the yield curve (stored in model.y) and the
% dividend yield curve (stored in model.q).

% The default interpolation method is 'flat'. You can also change this to
% 'linear' if you wish. An example (for a yield curve):
y_vals = [0.01;0.02;0.023;0.025;0.0255];
yield_curve = CurveClass('gridpoints',t_pts,'values',y_vals,...
                         'interpolation','linear');

y_eval = yield_curve.Eval(t_eval);
figure;
plot(t_eval,y_eval);
xlabel('Expiry');ylabel('Yield');title('Yield curve');
ylim([0,max(y_eval)*1.1]);

% As can be seen, the extrapolation is however still flat.

%% Modifying the settings of the pricing algorithm:
% Here we explain how one can change the settings of the pricing algorithm.
% The explanations given here will however be brief - you should consult
% the code for the details.

% The settings are stored here:
settings = model.pricerSettings

% The following parameters specify how many steps to simulate per year for 
% different groups of expiries. As an example the value settings.n(1)
% specifies how many steps per year should be used to price expiries from 0 to 
% settings.tn(1). The value settings.n(2) then specifies how many steps per
% year should be used to price expiries from settings.tn(1) to settings.tn(2).
% And so on...
model.pricerSettings.tn
model.pricerSettings.n

% In the following parameter we specify how many paths to simulate for 
% Monte Carlo pricing (this includes any antithetic paths):
model.pricerSettings.N

% In this sub-property we specify any special techniques to use in
% the price estimation:
model.pricerSettings.price_estimation

% Various control variates are available here. Options are 'asset_price'
% and 'none':
model.pricerSettings.price_estimation.control_variate = 'asset_price';

% We can also choose if we want to use antithetic paths:
model.pricerSettings.price_estimation.antithetic = true;

% And if we want to use conditional Monte Carlo (must be false as 
% feature is not supported for the extended rough Bergomi model):
model.pricerSettings.price_estimation.conditional_monte_carlo = false;

% From the put-call parity we know that it in theory does not matter if we
% compute the put or call option price (the other can always be obtained
% from the parity relation). However when computing the Monte Carlo estimator, 
% the standard errors may not be the same if we estimate the time value using a
% put or call option. With the 'option_type' parameter we set which option
% type we use to estimate the time value. Options are 'call','put' and 'otm'.
% The recommended choice is the default value of 'otm' (out-of-the-money):
model.pricerSettings.price_estimation.option_type = 'otm';

% If you wish to use more paths than you have RAM available for in a single
% computation you can do something else. This parameter specifies how many
% times to rerun the pricing algorithm and average the results:
model.pricerSettings.nRepeat = 10;
model.pricerSettings.N = 10000;

% The settings chosen above therefore specify that prices will be computed using
% 100,000 paths in total by running the pricing algorithm 10 times with
% 10,000 paths each and then averaging. Be aware that with some 
% price_estimation specifications each run may get increasingly biased as 
% settings.N -> 0 even though settings.nRepeat is increased comparatively. 
% It is therefore recommended to not choose settings.N too low. This is also to 
% avoid biasing the computation of standard errors. Standard errors are 
% aggregated as
%   
%   average standard error over settings.nRepeat runs / sqrt(settings.nRepeat)  
%
% to obtain the standard error of the averaged price.

% Reset setting:
model.pricerSettings.nRepeat = 1;

% With the 'precision' property we can decide in which precision the
% computations are to be done. Options are 'double' and 'single'. In
% general it is only the most heavy computations that are done in single
% precision if that is specified. 

% Let us briefly illustrate the speed up:
k = (-0.3:0.05:0.15)'; 
T = [0.05;0.2;0.5;1;2;3]; 

model.pricerSettings.precision = 'single';
tic;
model.GetPrices(k,T,true);
toc;

model.pricerSettings.precision = 'double';
tic;
model.GetPrices(k,T,true);
toc;

% The other properties (N_vol_grp, random and randomMaxT) are generaly 
% not to be used by an end-user.

