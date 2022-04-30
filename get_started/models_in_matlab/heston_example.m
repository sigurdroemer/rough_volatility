%% Clear workspace and load code:
clear;
project_folder = fileparts(fileparts(fileparts(matlab.desktop.editor.getActiveFilename)));
addpath(genpath(project_folder));

%% Define a Heston model (and inspect the object)
model = HestonClass('kappa',2,'vbar',0.2^2,'v0',0.2^2,'eta',2,'rho',-0.65,'s0',100);
model

% Curves are stored as CurveClass objects:
model.y
model.q

% Note how the yield curve (model.y) and dividend yield curve (model.q) by default 
% are assumed flat at zero. The forward variance curve (model.xi) is set to the fixed 
% value previously inputted. 

%% Compute implied volatility:
% Log-moneyness values, i.e. log(strike/forward):
k = (-0.3:0.05:0.2)'; 

% Expiries:
T = [0.05;0.2;0.5;1;2;3]; 

% Return prices for the cartesian product of the inputs:
cartProd = true;

% Compute prices:
ivSurf = model.GetPrices(k,T,cartProd);
ivSurf

% Plot:
for i=1:size(ivSurf.k,2)
    plot(ivSurf.k(:,i),ivSurf.iv(:,i),'-o','DisplayName',...
         ['T = ',num2str(T(i))],'LineWidth',1.25);
    hold on;
end
hold off;
title('Heston smiles');
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
title('Put option prices under Heston');
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
idxCall

% The input vectors can also be interpreted row-by-row like this:
k = [0;-0.2;0;0.1;0.15];
T = [0.1;0.1;0.2;0.2;0.5];
cartProd = false;
p = model.GetPrices(k,T,cartProd,'priceType','implied_volatility')

%% Settings:
% It is possible to change some of the settings of the pricing algorithm. More
% information can be found by reading the description of the HestonClass
% and FourierPricerSettingsClass.
model.pricerSettings
