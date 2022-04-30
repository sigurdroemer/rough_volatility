# Implementation of rough volatility models
This project implements the pricing models used in part one of the analysis of [1] as well as fast neural network approximations of these.

We start by outlining the models: Let S(t) denote the time t price of an asset and let r(t) and q(t) denote the risk-free interest rate and the continuously compounded dividend yield respectively; r(t) and q(t) are assumed deterministic. Under some standard assumptions, we then have 

![dS_eqn](https://latex.codecogs.com/svg.image?dS_t%26space%3B%3D%26space%3BS_t%28r%28t%29-q%28t%29%29dt%26space%3B%26plus%3B%26space%3BS_t%26space%3B%5Csqrt%7BV_t%7DdW_%7B2%2Ct%7D)

under the risk-neutral measure. Here V(t) is some process (the instantaneous variance) and we write W for Brownian motions.

The models for V(t) that we consider are defined below.

**Remark:** The notation differs slightly between the paper and the code. Keep this in mind.

### Heston
In the Heston model of [2], we have

![](https://latex.codecogs.com/svg.image?dV_t&space;=&space;\kappa(v_{\infty}-V_t)dt&space;&plus;&space;\eta&space;\sqrt{V_t}dW_{1,t})

![](https://latex.codecogs.com/svg.image?%5Ctext%7Bwhere%26space%3B%7D%26space%3B%5Ckappa%2C%5Ceta%2Cv_%7B%5Cinfty%7D%2CV_0%26space%3B%5Cgeq%26space%3B0%2C%26space%3B%5Ctext%7B%26space%3Band%26space%3B%7D%26space%3BdW_%7B1%2Ct%7DdW_%7B2%2Ct%7D%26space%3B%3D%26space%3B%5Crho%26space%3Bdt%2C%26space%3B%5Crho%26space%3B%5Cin%26space%3B%5B-1%2C1%5D)

### Rough Heston
We consider also a rough Heston model akin to [3] defined by

![rHeston](https://latex.codecogs.com/svg.image?V_t%26space%3B%3D%26space%3B%5Cxi_0%28t%29%26space%3B%26plus%3B%26space%3B%5Cfrac%7B%5Cnu%7D%7B%5CGamma%28H%26plus%3B%5Cfrac%7B1%7D%7B2%7D%29%7D%5Cint_0%5Et%26space%3B%28t-s%29%5E%7BH-%5Cfrac%7B1%7D%7B2%7D%7D%5Csqrt%7BV_s%7DdW_%7B1%2Cs%7D%2C%26space%3B%5Cphantom%7Bxx%7D%26space%3Bt%26space%3B%5Cgeq%26space%3B0%2C)

![](https://latex.codecogs.com/svg.image?%5Ctext%7Bwhere%26space%3B%7D%26space%3B%5Cnu%26space%3B%5Cgeq%26space%3B0%2C%26space%3B%5Cphantom%7Bx%7DH%26space%3B%5Cin%26space%3B%280%2C1%2F2%29%2C%26space%3B%5Cphantom%7Bx%7D%26space%3BdW_%7B1%2Ct%7DdW_%7B2%2Ct%7D%26space%3B%5Ctext%7B%26space%3Bfor%26space%3B%7D%26space%3B%5Crho%26space%3B%5Cin%26space%3B%5B-1%2C1%5D%2C%26space%3B%5Ctext%7B%26space%3Band%26space%3B%7D%26space%3B%5Cxi_0%26space%3B%5Ctext%7B%26space%3Bis%26space%3Bof%26space%3Bthe%26space%3Bform%26space%3B%7D)

![rHeston](https://latex.codecogs.com/svg.image?%5Cxi_0%28t%29%26space%3B%3D%26space%3BV_0%26space%3B%26plus%3B%26space%3B%5Cfrac%7B1%7D%7B%5CGamma%28H%26plus%3B%5Cfrac%7B1%7D%7B2%7D%29%7D%5Cint_0%5Et%26space%3B%28t-s%29%5E%7BH-%5Cfrac%7B1%7D%7B2%7D%7D%26space%3B%5Ctheta%28s%29%26space%3Bds%2C%26space%3B%5Cphantom%7Bxx%7D%26space%3Bt%26space%3B%5Cgeq%26space%3B0%2C)

### Rough Bergomi
The rough Bergomi model of [4] assumes

![rBergomi](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/rbergomi_2.png)

where

![](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/image3.png)

### Extended rough Bergomi
The extended rough Bergomi model assumes

![rBergomiExt](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/rbergomi_ext_def.png)

where

![rBergomiExtFactors](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/rbergomi_ext_vfactors.png)

and

![](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/image4.png)

It is however natural to reexpress it in terms of the following parameters:

![](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/rho_eta_rbergomi_ext.png)

This is also the parameterization used in the code.

## What the code contains
The code first and foremost implements pricing algorithms for puts and calls on S(t). You should consult the code or the paper for a description of **what** methods and schemes are used. This part of the project is only implemented in Matlab.

Secondly, there are Matlab scripts for generating large datasets of option prices for different model parameters. Neural networks are then trained (in Python and using Keras) to represent the datasets for each model.

Thirdly, the code implements interfaces to evaluate the neural networks. The interfaces are available in Matlab, Python and R. 

The network weights are located in the folder *.../code/neural_networks/data/neural_network_weights* if one wants to implement the neural networks in other languages. The training and test datasets are rather large (almost 30 GB in total) and can thus instead be downloaded from [here](https://drive.google.com/open?id=1dmWpm8d5l6yrYyv2twcHMnMmh5B1QaP_). To use the datasets in the project, place the files in the folder *".../code/neural_networks/data/training_and_test_data".

### Speed and accuracy
With neural networks we can compute an entire volatility surface in around 1 millisecond on a standard laptop. A typical calibration to SPX options can then be performed in less than a second. Read the paper for the details, or try it yourself.

The networks are also highly accurate as exemplified with the rough Bergomi model below:

![Explot](get_started/neural_networks_in_matlab/example_plot.jpg?raw=true "Title")

The parameters are: 

![](https://latex.codecogs.com/svg.image?H&space;=&space;0.1,&space;\phantom{x}\eta&space;=&space;2.1,&space;\phantom{x}\rho&space;=&space;-0.9,&space;\phantom{x}\xi_0(t)&space;=&space;0.15^2.)

A more detailed analysis of the approximation error can be found in the paper.

## Getting started
There are a number of scripts to help get you started. They are explained below.

| Language        | Folder        | Description  |
| :--------------- |:-------------| :------------|
| Matlab          | .../get_started/neural_networks_in_matlab      | Examples of using the neural networks in Matlab. |
| Python          | .../get_started/neural_networks_in_python      | Examples of using the neural networks in Python. |
| R               | .../get_started/neural_networks_in_R           | Examples of using the neural networks in R.|
| Matlab          | .../get_started/models_in_matlab | Examples of using the underlying pricing models in Matlab. |

Remarks: 
- The Matlab code was developed in version 2019a, the R code in version 3.4.3 and the Python code in version 3.7.1. There is no guarantee that the code will work in older versions.
- Only the Matlab version of the neural network implementations have been optimised for speed. Thus this is the recommended version (although they are all fast).

## Main references
1. Rømer, S.E., Empirical analysis of rough and classical stochastic volatility models to the SPX and VIX markets, 2022, to be published soon.
2. Heston, S. L., A closed-form solution for options with stochastic volatility with applications to bond and currency options. Review of Financial Studies 6, 1993, pp. 327–343.
3. El Euch, O., Gatheral, J., and Rosenbaum, M., Roughening Heston. Risk, May 2019, pp. 84-89.
4. Bayer, C., Friz, P., and Gatheral, J., Pricing under rough volatility. Quantitative Finance, 2016, 16(6), pp. 887-904.
5. Horvath, B., Muguruza, A. and Tomas, M., Deep learning volatility: A deep neural network perspective on pricing and calibration in (rough) volatility models. Quantitative Finance, 2021, 21(1), pp. 11-27.

## External packages and libraries
The following external packages and libraries are included in the project:
- Adi Navve (2020). Pack & Unpack variables to & from structures with enhanced functionality (https://www.mathworks.com/matlabcentral/fileexchange/31532-pack-unpack-variables-to-from-structures-with-enhanced-functionality), MATLAB Central File Exchange. Retrieved March 16, 2020.

## Other remarks:
- The neural network datasets for the rough Bergomi models are computed with [this file](https://drive.google.com/drive/folders/1QRv71nhHvZ_rB0kPjO3cHpFOE3GVHn9J) in place of the *HybridScheme.m* file of this project. The only difference between the files is the FFT implementation, which has been made more readable in the current project. The two implementations are equivalent up to round-off error.
