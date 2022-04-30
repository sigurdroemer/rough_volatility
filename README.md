# Implementation of rough volatility models
This project implements the stochastic volatility models used in [1] as well as fast neural network approximations of these.

Let us start by explaining the models we work with. Let therefore S(t) denote the time t price of some (underlying) asset and let r(t) and q(t) denote the risk-free interest rate and the dividend yield respectively (both assumed deterministic functions of time). Under some standard assumptions we then have

![dS](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/dS_2.png)

under the risk-neutral measure. Here V(t) is the instantaneous variance process and we let W's denote Brownian motions.

In this project we then consider the (risk-neutral) models for V(t) described below. One of them (Heston) is a non-rough volatility model, the other three are proper rough volatility models.

### Heston
The Heston model of [2] assumes

![heston](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/heston_2.png)

where 

![](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/image1.png)

### Rough Heston
The rough Heston model of [3] assumes

![rHeston](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/rheston_2.png)

where

![](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/image2.png)

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
The code first and foremost implements pricing algorithms for pricing put and call options on S(t) for each of the four models explained above. You should consult the code or the paper for a description of **what** methods and schemes are used. This part of the project is only implemented in Matlab.

Secondly, there are Matlab scripts for generating large datasets of option prices for different model parameters. Neural networks are then trained (in Python and using Keras) to represent the datasets for each model.

Thirdly, the code implements an interface for evaluating the neural networks for arbitrary model parameters (within the trained domain). The neural network interfaces are available in Matlab, Python and R. 

The network weights are located in the folder *.../code/neural_networks/data/neural_network_weights* if one wants to implement the neural networks in other languages. The training and test datasets are rather large (almost 30 GB in total) and can thus instead be downloaded from https://drive.google.com/open?id=1S7C8T7ak1_pTjeffC-Oynnjewxojyemp. To use the datasets in the project, place the files in this folder: *".../code/neural_networks/data/training_and_test_data"*.

### Speed and accuracy
With a neural network approximation one can compute an entire implied volatility surface in around 1 millisecond on a standard laptop. A typical calibration to SPX option prices can then be done in less than a second. Read the paper for the details (or try it yourself).

The networks are also highly accurate as exemplified with the rough Bergomi model below:

![](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/example_plot.jpg)

The parameters are: 

![](https://github.com/sigurdroemer/rough_volatility/blob/readme_images/parameters.JPG)

A more detailed analysis of the approximation error can be found in the paper.

## Getting started
There are a number of scripts to help get you started. They are explained below.

| Language        | Folder        | Description  |
| :--------------- |:-------------| :------------|
| Matlab          | .../get_started/neural_networks_in_matlab      | Examples of using the neural network models in Matlab. |
| Python          | .../get_started/neural_networks_in_python      | Examples of using the neural network models in Python. |
| R               | .../get_started/neural_networks_in_R           | Examples of using the neural network models in R.|
| Matlab          | .../get_started/models_in_matlab | Examples of using the underlying pricing models in Matlab. |

Remarks: 
- The Matlab code was developed in version 2019a, the R code in version 3.4.3 and the Python code in version 3.7.1. There is no guarantee that the code will work in older versions.
- Only the Matlab version of the neural network implementations have been optimised for speed. Thus this is the recommended version (although they are all fast).

## Main references
1. Rømer, S.E., Historical analysis of rough volatility models to the SPX market, ???
2. Heston, S. L. (1993). “A closed-form solution for options with stochastic volatility with applications to bond and currency options”. In: Review of Financial Studies 6, pp. 327–343.
3. Euch, O. E., J. Gatheral, and M. Rosenbaum (2019). “Roughening Heston”. In: Risk, pp. 84-89, May 2019.
4. Bayer, C., P. Friz, and J. Gatheral (2016). “Pricing under rough volatility”. In: Quantitative Finance, 16(6):887-904.
5. Horvath, B., A. Muguruza, and M. Tomas (2019). “Deep Learning Volatility”. Working paper. Available at https://ssrn.com/abstract=3322085 (accessed 16th of June 2020).

## External packages and libraries
The following external packages and libraries are included in the project:
- Adi Navve (2020). Pack & Unpack variables to & from structures with enhanced functionality (https://www.mathworks.com/matlabcentral/fileexchange/31532-pack-unpack-variables-to-from-structures-with-enhanced-functionality), MATLAB Central File Exchange. Retrieved March 16, 2020.

## Updates:
16-June-2020 (commit number = f2c95a6):  Updated HybridScheme.m by improving code readability of FFT convolution. Implementation is equivalent to the previous version (commit number f2bad0f) up to round-off error. Test and training datasets are still computed under commit number f2bad0f.
