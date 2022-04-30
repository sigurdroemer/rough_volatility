function [out_data,out_header,inputFail,errFail] = ComputeTrainingData(model,...
                                                          nSamples,params,...
                                                          samplingSettings,...
                                                          k,T,varargin)
% Description: Computes option prices for a pricing model for many different
% input parameters. This can then be used to train a neural network 
% representation. 
% 
% Remarks: 
%   o Prices are computed for call options. One will have to
%     post-process the data if e.g. implied volatility is desired.
%
%   o Standard errors will also be returned if model.pricerSettings 
%     is a 'MonteCarloPricerSettingsClass'.
%
%   o We assume that the input model has zero interest rates and dividends. 
%     If this is not the case we automatically overwrite the input model.
%
% Parameters:
%   model:    [1x1 PricingModelClass] Model to generate samples from.
%
%   nSamples: [1x1 integer] Number of samples to generate.
%
%   params:   [1xN cell] Cell array of strings with the names of the
%             parameters to be varied when computing different samples. The
%             names must exactly correspond to the object properties of the
%             model input. The parameters/properties must either be
%             scalars or CurveClass objects. The params vector must not
%             contain any duplicate entries.
%
%   samplingSettings:
%             [1xN cell] Each element must be a [1x1 struct] with settings
%             detailing how the given parameter should be sampled. In the
%             following it will be explained which fields each [1x1 struct]
%             can and/or should have and what effect each field has.
%
%             However before that is explained it is important to
%             understand that some samples may fail to execute properly.
%             Examples are if the convexity and Merton's tunnel checks fail
%             or if the numerical scheme fails to converge etc. (can happen
%             when pricing very far out-of-the-money options). Thus: For
%             the first attempt a number nSamples of input parameters are
%             sampled according to the samplingSettings. Then these are
%             attempted to be priced. Say a number nFail of them fails to
%             execute properly. Then another batch of nFail input
%             parameters are sampled according to the samplingSettings.
%             This continues until there are no more failed samples.
%             Because of all this the final set of samples may deviate
%             some from what was intended - all depending on how often
%             samples fail to execute properly and if there are certain
%             parts of the parameter space where this is more likely to
%             happen.
%
%             Now the required [1x1 struct]'s will be explained: 
%
%             First of all there should be fields 'lb' and 'ub' each of which 
%             should be set to scalar values. These values then set the lower 
%             and upper bounds of the sampling distribution for that particular
%             parameter.
%
%             Secondly there must be a field 'method' which must be set to 
%             a string. This determines the general method for sampling this 
%             particular parameter. The possible options are explained below:
%
%               o  'unif_cont': Parameter is sampled from a continuous
%                  uniform distribution on the interval between 'lb' and 'ub'.
%                  No further fields are required to be set.
%
%               o  'unif_discrete': Parameter is sampled from a discrete
%                  uniform distribution. The grid points are chosen
%                  equidistantly between 'lb' and 'ub'. The number of intervals 
%                  between 'lb' and 'ub' should be specified with a field named 
%                  'n'. Thus there will be n + 1 grid points / values that
%                  can be sampled.
%
%               o  'trunc_norm': Parameter is sampled from a truncated
%                  normal distribution. The distribution is truncated
%                  to the interval from 'lb' to 'ub'. The mean and standard 
%                  deviation (of the non-truncated version) should then be 
%                  set in the fields 'mu' and 'sig'.
%
%               o  'unif_discrete_random_grid': Similar to 'unif_discrete'
%                  except the grid points / attainable values are random.
%                  This works as follows: The field 'n' should again be
%                  set to a positive integer. Then letting dx = (ub-lb)/n
%                  we on each interval
%
%                  [lb+(i-1)*dx, lb+i*dx] for i=1,...,n
%
%                  sample a random grid point from a continuous uniform
%                  distribution. The final parameter(s) will then be
%                  sampled from a discrete uniform distribution over these
%                  (randomly sampled) grid points.
%
%                  The field 'resampleEvery' should also be set to a
%                  positive integer. The (random) grid will then be
%                  resampled once every 'resampleEvery' input sample. 
%               
%                  This method does not support the CurveClass type.
%
%               o  'trunc_norm_discrete_random_grid': Similar to
%                  'trunc_norm' except the distribution is discrete with
%                  grid points that are sampled randomly. In more detail the 
%                  sampling works as follows: First of all there should be an 
%                  additional field 'n' set to a positive integer. Then letting 
%                  dx = (ub-lb)/n we on each interval
%
%                   [lb+(i-1)*dx, lb+i*dx] for i=1,...,n
%
%                  sample a random grid point from a continuous uniform 
%                  distribution.
%                 
%                  The fields 'mu' and 'sig' should also be set to the mean
%                  and standard deviation of a truncated normal distribution.
%                  We then sample the parameter(s) according to a discrete
%                  distribution with probability masses
%
%                  p_i = G(lb+i*dx) - G(lb+(i-1)*dx) for i=1,...,n
%
%                  in each (random) grid point. Here G() denotes the
%                  cumulative distribution function of the truncated normal
%                  distribution with bounds 'lb' and 'ub' as well as mean and
%                  standard deviation 'mu' and 'sig'.
%
%                  A field named 'resampleEvery' should also be set to a positive 
%                  integer. The (random) grid will then be resampled once every 
%                  'resampleEvery' parameter sample. 
%
%                  This method does not support the CurveClass type.
%
%               o  'xi_special': This method should only be used if the 
%                  parameter is named 'xi', is of the CurveClass type and is 
%                  interpreted as a forward variance curve. The (additional) 
%                  required  fields are 'Ts','perc','vol0_low','vol0_high',
%                  'vs_rate_low','vs_rate_high','kappa_low','kappa_high', 
%                  'eps_noise','num_flat_sections','type'. These exactly correspond 
%                  to those needed for a call to the function SampleCurves from 
%                  the class ForwardVarianceCurveSamplingClass. See the description 
%                  of that function for the required dimensions and an 
%                  interpretation of the mentioned fields. The input
%                  xi_bounds to that function will then be set to [lb,ub]
%                  and the forward variance curves are then sampled
%                  according to the SamplesCurves function. 
%
%               o  'theta_special': This should only be used to sample theta
%                   curves for the rough Heston model (class 'rHestonClass').
%                   Theta-curves are sampled by sampling first xi-curves using 
%                   the method 'xi_special' - thus the fields that apply (and 
%                   should be used) are exactly those of the 'xi_special' method. 
%                   It should be noted that v0 will also be sampled as part of
%                   the supplied forward variance maturities. Thereafter a piecewise 
%                   constant theta curve (and v0) is chosen such that the resulting 
%                   forward variance curve matches the sampled points on the 
%                   xi-curve. If the non-negativity requirement (see function
%                   'CheckNonNegReqTheta') is not satisfied, we keep resampling
%                   xi (i.e. (v0,theta)) for the given other parameters, until 
%                   the requirement is satisfied. If this method is used, the 
%                   variables 'H' and 'v0' must also be sampled and they must be 
%                   placed before 'theta' in the params array. It should here be 
%                   noted that the 'v0'-values are overwritten and set equal to 
%                   the shortest maturity forward variance that is sampled.
%
%             In addition, all methods allow for a function to be applied 
%             after the sampling as described above has been completed. This 
%             should be specified by setting another field named 
%             'post_sampling_transformation' equal to a [1x1 function]
%             which we can call F. Thus after sampling the inputs as previously 
%             explained, the function F will then be applied to the result.
%             This function should be able to take a general matrix of inputs  
%             and return a matrix of the same size. If this field does not exist 
%             (or is empty) no transformation will be applied.
%           
%   k:        [Mx1 real] Log-moneyness values to generate samples for.
%
%   T:        [Mx1 real] Expiries to generate samples for.
%
% Optional parameters (inputted as name-value pairs):
%   getInputsOnly: 
%           [1x1 logical] If set to true only the parameter input matrix is 
%           returned. Useful for inspecting the parameter sampling before 
%           computing any prices. Default is false. See the description
%           of the output for more details. 
%
%   use_sobol: 
%           [1x1 logical] If true we use the Sobol sequence to fill out 
%           the sampling space. Default is false. Important: The use of the 
%           Sobol sequence does not apply to parameters where the sampling 
%           method is set to 'xi_special'. Here pseudo random numbers are 
%           always used no matter what value of use_sobol is chosen.
%
%   nWorkers: 
%           [1x1 integer] Specifies how many parallel workers to use to compute 
%           the samples. Default is 0 (not in parallel).
%
%   saveEvery: 
%           [1x1 integer] Specifies how many samples to compute before saving 
%           the temporary results. Very useful to recover some of the already 
%           completed computations if something goes wrong while computing 
%           (i.e. the system shuts down etc.). Default is Inf in which case we 
%           only save the results each time we have made a full attempt at
%           computing all nSamples.
%
%   tmpFolder: 
%           [1x1 string] Specifies where to save the temporary results and 
%           states. Only needed if saveEvery < Inf. Folder should be empty 
%           except for the temporary files that will be generated by this 
%           function.
%
%   folderSlash: 
%           [1x1 string] The way to specify subfolders may depend on the system 
%           used. To save the temporary results the function needs to know how 
%           the folders are separated. The options are '\' and '/'. The 
%           default is '\'.
%
%   checkMertonsTunnel: 
%           [1x1 logical] If set to true then if there are any prices in a 
%           computed sample that do not satisfy Merton's tunnel we disregard 
%           the entire sample.
%
%   checkConvexity: 
%           [1x1 logical] If true we check prices for convexity in the strike 
%           dimension. If the check fails anywhere for a given sample we 
%           discard the entire sample. Default is true.
%
%   reuse_numbers_for_each_H: 
%           [1x1 logical] Special parameter only to be used by the 
%           'rBergomiClass' model. If set to true we reuse the random numbers 
%           for all runs where the value of the Hurst parameter is the same. 
%           This generally saves a lot of time. To achieve any speed improvement  
%           it should however be combined with one of the sampling methods
%           'unif_discrete', 'unif_discrete_random_grid' or
%           'trunc_norm_discrete_random_grid'.
%
%   reuse_numbers_for_each_alpha:
%           [1x1 logical] Special parameter only to be used by the 
%           'rBergomiExtClass' model. Works the same as the 
%           reuse_numbers_for_each_H parameter except for the 'alpha' parameter 
%           of the 'rBergomiExtClass'.
%
%   seed:   [1x1 integer] Seed for generating random numbers. If empty, no
%           seed is set. If set, we use the generator 'mrg32k3a' with
%           substreams to ensure reproducability even if samples are run in
%           parallel. If a seed is inputted we restore the random stream to
%           its original type and state once the algorithm finishes (or if it
%           stops because of an error). If each price estimation is
%           deterministic it is recommended to instead set the seed outside this 
%           function call.
%
% Output: 
%   If getInputsOnly = false (the default) the outputs are as follows:
%
%   out_data:   [nSamples x nTotalCols real] Matrix consisting of input 
%               parameters, call option prices and (if applicable) standard 
%               errors and that only for the samples that executed without any 
%               errors.
%
%   out_header: [1 x nTotalCols cell] Header corresponding to 'out_data'.
%
%   inputFail:  [nFail x nInput real] Input parameters of the say nFail samples 
%               that failed to execute properly. Columns correspond to the first
%               nInput < nTotalCols entries in 'out_header'.
%
%   errFail:    [nFail x 1 cell] Error messages for each of the nFail failed 
%               samples. A non empty message is only shown if an actual error
%               was thrown when attempting to compute the prices and not if a 
%               sample is categorized as failed because it did not satisfy the 
%               convexity or Merton's tunnel checks.
%
%   If getInputsOnly = true we will instead have the following: (1) The 
%   'out_data' matrix will now consists of the sampled input parameters. It 
%   will then be of the size nSamples x nInput. Also, 'out_headers' will then 
%   be of the size 1 x nInput giving the name of each column in 'out_data'. 
%   The remaining outputs will be empty. Note also that since some inputs may 
%   fail to execute properly (or otherwise be discarded) if 
%   getInputsOnly = false the set of input parameters given may differ from
%   those returned in that case.
%

% Parse and validate inputs:
p = inputParser;
addOptional(p,'getInputsOnly',false);
addOptional(p,'reuse_numbers_for_each_H',false);
addOptional(p,'reuse_numbers_for_each_alpha',false);
samplingMethodDefault = cell(size(params));
for i=1:size(params,2);samplingMethodDefault{i} = 'unif_cont';end
addOptional(p,'nWorkers',0);
addOptional(p,'saveEvery',0);
addOptional(p,'use_sobol',false);
addOptional(p,'tmpFolder',[]);
addOptional(p,'folderSlash','\');
addOptional(p,'checkMertonsTunnel',true);
addOptional(p,'checkSmileConvexity',true);
addOptional(p,'seed',[]);
parse(p,varargin{:});
v2struct(p.Results);

model.y = CurveClass('gridpoints',1,'values',0);
model.q = CurveClass('gridpoints',1,'values',0);

if saveEvery < 1
    error('GetTrainingData: Parameter saveEvery must not be less than 1.');
end

if ~isempty(tmpFolder);mkdir(tmpFolder);end

% Initialize outputs:
[out_data,out_header,inputFail,errFail] = deal([]);

if any(size(dir([tmpFolder '/*.mat' ]),1))
      disp(['ComputeTrainingData: Preexisting (.mat) files were found in the ',...
            'temporary folder given.']);
      disp(['Type true (or 1) to continue', ...
          ' in which case we will continue from ',...
          ' the last known state.']);
      disp(['Type false ',...
          '(or 0) to cancel this function call',...
          ' and return to the workspace.']);
    continue_computations = input('');
    if ~continue_computations
        disp('ComputeTrainingData: Function call cancelled.');
        return;
    end
end

if saveEvery < Inf
    disp(['Temporary results are saved to the folder: ', tmpFolder]);
end

if any(model.y.values ~= 0) || any(model.q.values ~= 0) ...
        || any(strcmpi(params,'y')) || any(strcmpi(params,'q'))
    error(['ComputeTrainingData: Function assumes the model given has zero ',...
          'interest rate and zero dividend.']);
end

if reuse_numbers_for_each_H && ~strcmpi(class(model),'rBergomiClass')
    error(['ComputeTrainingData: Parameter ''reuse_numbers_for_each_H'' ',...
          'can only be set to true if the model is a ''rBergomiClass''.']);    
end

if reuse_numbers_for_each_alpha && ~strcmpi(class(model),'rBergomiExtClass')
   error(['ComputeTrainingData: Parameter ''reuse_numbers_for_each_alpha'' ',...
          'can only be set to true if the model is a ''rBergomiExtClass''.']);    
end

[randomSeedArray,states,origRNG,globalStream] = deal([]);
if ~isempty(seed)
    % Set the substreams (the first one is for sampling the parameters, the
    % rest for evaluating the samples):
    nStreams = 1 + saveEvery;
    randomSeedArray = RandStream.create('mrg32k3a','NumStreams',nStreams,...
                                        'Seed',seed,'CellOutput',true);
    states = cell(size(randomSeedArray));
    origRNG = rng;
    globalStream = RandStream.getGlobalStream;
end

try
    % Count true number of input parameters and create header:
    [headers, nParamTotal] = ConstructInputHeader(model,params);
    
    % Generate (initial) input samples:
    if ~isempty(seed)
        RandStream.setGlobalStream(randomSeedArray{1});
    end
    
    [input_new, idxSplit] = GenNewSamples(1,nSamples,nParamTotal,params,...
            samplingSettings,model,use_sobol,reuse_numbers_for_each_H,...
            reuse_numbers_for_each_alpha,saveEvery);
        
    if ~isempty(seed)
        states{1} = globalStream.State;
    end
        
    % Keep track of the total number of samples generated so far (incl. failed 
    % ones):
    nGen = nSamples;

    % Return early if only the inputs were requested:
    if getInputsOnly
        out_data = input_new;
        out_header = headers;
        if ~isempty(origRNG);rng(origRNG);end
        return;
    end
    
    % Define output table:
    headers = GetFullHeader(headers,size(k,1));

    % Set up parallel pool:
    if ~isempty(gcp('nocreate'))
        delete(gcp('nocreate'));
    end
    if nWorkers > 0
        parpool('local',nWorkers);
    end
    
    % Start outer loop
    [input_par,output,output_stderr] = deal([]);
    finished = false;iterSave = 1;
    while ~finished
         
        % Loop over each section of inputs:
        nFailed = 0;
        for j=1:size(idxSplit,1)-1
            
            fn = ['temp','_',num2str(iterSave),'.mat'];
            if ~isempty(tmpFolder) ...
                    && exist([tmpFolder,folderSlash,fn], 'file') == 2
                % Attempt to load results from folder
                load([tmpFolder,folderSlash,fn]);
                iterSave = iterSave + 1;
            else
                % Else evaluate samples:
                input_sub = input_new(idxSplit(j)+1:idxSplit(j+1),:);
                [output_new,output_stderr_new,errs_new,states] = EvaluateSamples(...
                    model,input_sub,params,k,T,nWorkers,...
                    reuse_numbers_for_each_H,...
                    reuse_numbers_for_each_alpha,...
                    randomSeedArray,states);
                
                if ~isempty(tmpFolder)
                    % Store results on drive if required:
                    save([tmpFolder,folderSlash,fn],'output_new',...
                                'output_stderr_new','errs_new','input_sub');
                    iterSave = iterSave + 1;
                end
            end
            
            % Remove NaN or zero rows and replace them with new ones:
            idxFailed = any(output_new<=0,2) | any(isnan(output_new),2);
            % Check Mertons tunnel:
            if checkMertonsTunnel
                idxFailedMerton = any(output_new > model.s0 | ...
                    bsxfun(@ge,max([model.s0 - model.s0.*exp(k),...
                    zeros(size(k))],[],2)',output_new),2);
                idxFailed = idxFailed | idxFailedMerton;
            end
            % Check prices for convexity:
            if checkSmileConvexity
                for i=1:size(output_new,1)
                    if ~idxFailed(i) && ...
                           CheckConvexity(output_new(i,:)',k,T,model.s0)
                        idxFailed(i) = true;
                    end
                end
            end
            
            nFailed = nFailed + sum(idxFailed);

            % Store failed results:
            inputFail = [inputFail;input_sub(idxFailed,:)];
            errFail = [errFail;errs_new(idxFailed)];

            % Add results to existing:
            input_par = [input_par;input_sub(~idxFailed,:)];
            output = [output;output_new(~idxFailed,:)];
            output_stderr = [output_stderr;output_stderr_new(~idxFailed,:)];

            disp([num2str(size(input_par,1)), ' samples out of ', ...
                  num2str(nSamples), ' generated succesfully']);
            disp(['Number of failed is ', num2str(size(inputFail,1))]);
        end
        
        if nFailed == 0
            finished = true;
        else
            % Generate new input samples:
            if ~isempty(seed)
                RandStream.setGlobalStream(randomSeedArray{1});
                globalStream.State = states{1};
            end            
            
            nMissing = nSamples - size(input_par,1);
            [input_new, idxSplit] = GenNewSamples(nGen+1,nMissing,...
                                nParamTotal,params,samplingSettings,model,...
                                use_sobol,reuse_numbers_for_each_H,...
                                reuse_numbers_for_each_alpha,saveEvery);
            nGen = nGen + nFailed;
            
            if ~isempty(seed)
                states{1} = globalStream.State;
            end
            
        end
        
    end

    % Store results correctly:
    if isa(model.pricerSettings,'MonteCarloPricerSettingsClass')
        out_data = [input_par,output,output_stderr];
        out_header = headers;
    else
        out_data = [input_par,output];
        out_header = headers(1:size(k,1)+nParamTotal);
    end

    % Delete par-pool:
    delete(gcp('nocreate'));
    if ~isempty(origRNG);rng(origRNG);end
    
catch e
    % Delete par-pool:
    delete(gcp('nocreate'));
    if ~isempty(origRNG);rng(origRNG);end
    error(['GetTrainingData: An error occured: ',e.message]);
end
    
end

function [headers, nParamTotal] = ConstructInputHeader(model,params)
% Description: Constructs header for the input parameters and counts them.
    headers = {};nParamTotal = 0;
    for i=1:size(params,2)
        if isa(model.(params{i}),'CurveClass')
            nGridPts = size(model.(params{i}).gridpoints,1);
            nParamTotal = nParamTotal + nGridPts;
            for j=1:nGridPts
                headers = {headers{:},[params{i},'_',num2str(j)]};
            end
        else
            nParamTotal = nParamTotal + 1;
            headers = {headers{:},params{i}};
        end
    end
end

function header = GetFullHeader(headerInputsOnly,nOutput)

    header = headerInputsOnly;
    for i=1:nOutput
        header{size(header,2)+1} = ['price_',num2str(i)];
    end
    for i=1:nOutput
        header{size(header,2)+1} = ['se_',num2str(i)];
    end
    
end

function [input, idxSplit] = GenNewSamples(rowStart,nSamples,nParamTotal,...
                               params,samplingSettings,model,use_sobol,...
                               reuse_numbers_for_each_H,...
                               reuse_numbers_for_each_alpha,saveEvery)
% Description: Generates new samples.
%
% Parameters:
%   rowStart:    [1x1 integer] The row number to generate the first sample for.
%   nSamples:    [1x1 integer] The number of new samples to generate.
%   nParamTotal: [1x1 integer] The total number of input parameters.
%   params:      [1xN cell] Cell array of strings with name of every input
%                parameter. Here a CurveClass object counts as one input.
%
%   The remaining parameters are as described in the main function 
%   'GetTrainingData'.
%
% Output:
%   input:    [nSamples x M real] The new input samples.
%   idxSplit: [Lx1 real] Indices used for splitting up the samples for each
%             worker.
%

    % Precompute/extract the sobol numbers needed:
    if use_sobol
        
        % Determine if we need to also sample from the 'xi_special' or 
        % 'theta_special' distributions (these still use pseudo random
        % numbers regardless of how the parameter use_sobol is set):
        
        idx_xi_or_theta_special = false(size(params));
        for i=1:size(params,2)
           idx_xi_or_theta_special(i) = ...
                      strcmpi(samplingSettings{i}.method,'xi_special') || ...
                      strcmpi(samplingSettings{i}.method,'theta_special');
        end
        
        if sum(idx_xi_or_theta_special) == 0
            p = sobolset(nParamTotal);
        elseif sum(idx_xi_or_theta_special) == 1
            nParamXiOrTheta = size(model.(...
                          params{idx_xi_or_theta_special}).gridpoints,1);
            p = sobolset(nParamTotal - nParamXiOrTheta);
        else
            error(['GenNewSamples: Only one of the sampling methods ',...
                  '''xi_special'', ''theta_special'', can be used and that',...
                  ' only once and for a CurveClass type parameter with the ', ...
                  'name ''xi'', respectively, ''theta''.']);
        end
        U = p(rowStart:rowStart+nSamples-1,:);
        
    end
    
    % Initialize variables for loop:
    input = nan(nSamples,nParamTotal);
    iCol_1 = 0;
    
    % Loop over the parameters:
    for i=1:size(params,2)
        % Update the column index:
        iCol_1 = iCol_1 + 1;
        
        % Construct the post sampling transformation function:
        if ~isfield(samplingSettings{i},'post_sampling_transformation')
            F = @(x)(x);
        else
            F = samplingSettings{i}.post_sampling_transformation;
        end
        
        % Find the size of this particular parameter:
        if isa(model.(params{i}),'CurveClass')
            paramSize = size(model.(params{i}).gridpoints,1);
            curve_class = true;
        else
            paramSize = 1;
            curve_class = false;
        end
        
        iCol_2 = iCol_1 + paramSize - 1;
        
        lb = samplingSettings{i}.lb;
        ub = samplingSettings{i}.ub;
        method = samplingSettings{i}.method;
        
        % Find the code relevant for the sampling distribution/method chosen 
        % for this particular parameter:
        if (strcmpi(method,'unif_cont') || strcmpi(method,'trunc_norm'))

            % Get the uniform(0,1) random variables:
            if use_sobol
                u_sub = U(:,iCol_1:iCol_2);
            else
                u_sub = rand(nSamples,paramSize);
            end
            
            % Transform to the requested distribution:
            if strcmpi(method,'unif_cont')
                input(:,iCol_1:iCol_2) = F(lb + (ub-lb)*u_sub);
            elseif strcmpi(method,'trunc_norm')
                mu = samplingSettings{i}.mu;
                sig = samplingSettings{i}.sig;
                pd = truncate(makedist('Normal','mu',mu,'sigma',sig),lb,ub);
                input(:,iCol_1:iCol_2) = F(icdf(pd,u_sub));
            end
            
        elseif strcmpi(method,'unif_discrete')
            
            if use_sobol
                error(['GenNewSamples: Sobol numbers are currently not ',...
                       'supported when sampling from the distribution ',...
                       '''unif_discrete''.']);
            end
            if curve_class
                error(['GenNewSamples: It is currently not supported to ',...
                       'sample a parameter of the CurveClass type from the ',...
                       '''unif_discrete'' distribution.']);
            end
            scaleFactor = (ub - lb) ./ samplingSettings{i}.n;
            shiftFactor = lb - scaleFactor;
            u_sub = unidrnd(samplingSettings{i}.n + 1, nSamples, 1);
            input(:,iCol_1:iCol_2) = F(u_sub*scaleFactor + shiftFactor);
            
        elseif strcmpi(method,'xi_special')
            if ~curve_class || ~strcmpi(params{i},'xi')
                error(['GenNewSamples: To sample from the ''xi_special'' ',...
                       'distribution the parameter must be named ''xi'' ',...
                       'and be of the CurveClass type.']);
            end
            utils = ForwardVarianceCurveSamplingClass;
            settings = samplingSettings{i};
            input(:,iCol_1:iCol_2) = F(utils.SampleCurves(nSamples,...
                                 settings.Ts,settings.type,settings.perc,...
                                 settings.vol0_low,settings.vol0_high,...
                                 settings.vs_rate_low,settings.vs_rate_high,...
                                 settings.kappa_low,settings.kappa_high,...
                                 settings.eps_noise,[lb,ub],settings.num_flat_sections));

        elseif strcmpi(method,'theta_special')

            if ~curve_class || ~strcmpi(params{i},'theta')
                error(['GenNewSamples: To sample from the ''theta_special'' ',...
                       'distribution the parameter must be named ''theta'' ',...
                       'and be of the CurveClass type.']);
            end            
            
            idxV0 = find(strcmpi(params,'v0'));
            idxH = find(strcmpi(params,'H'));
            small_num = 10^(-6);
            
            utils = ForwardVarianceCurveSamplingClass;
            settings = samplingSettings{i};
                        
            idxValid = false(size(input,1),1);
            nSamplesDone = sum(idxValid);
            
            maxIter = 10^4;iter = 0;
            while nSamplesDone < nSamples
                % We keep resampling until we have a full set of curves that do not 
                % violate the non-negativity requirement.                
                
                iter = iter + 1;
                if iter > maxIter
                    error(['ComputeTrainingData:GenNewSamples: The maximum number of ',...
                           'iterations was reached while sampling (v0,theta).']);
                end

                % Sample:
                xi_temp = F(utils.SampleCurves(nSamples - nSamplesDone,...
                                     [small_num;settings.Ts],settings.type,settings.perc,...
                                     settings.vol0_low,settings.vol0_high,...
                                     settings.vs_rate_low,settings.vs_rate_high,...
                                     settings.kappa_low,settings.kappa_high,...
                                     settings.eps_noise,[lb,ub],settings.num_flat_sections)); 
                
                % Attempt to add samples to dataset:
                idxUpdate = find(~idxValid);
                for j=1:size(xi_temp,1)
                    % Get parameters:
                    v0 = xi_temp(j,1);
                    xi = xi_temp(j,2:end)';
                    H = input(idxUpdate(j),idxH);
                    
                    % Compute theta:
                    theta = GetThetaFromXi(v0,H,settings.Ts,xi)';
                    
                    % If non-negativity is ok, then add to dataset:
                    if CheckNonNegReqTheta(v0,H,settings.Ts,theta')
                        input(idxUpdate(j),idxV0) = v0;
                        input(idxUpdate(j),iCol_1:iCol_2) = theta;
                        idxValid(idxUpdate(j)) = true;
                    end

                end
                
                % Update counter:
                nSamplesDone = sum(idxValid);
                
            end
            
        elseif strcmpi(method,'unif_discrete_random_grid') ...
                || strcmpi(method,'trunc_norm_discrete_random_grid')

          if curve_class
              error(['GenNewSamples: Sampling method ', method,...
                    ' is not supported for parameters of the CurveClass type']);
          end
            
          dPar = (ub - lb) ./ samplingSettings{i}.n;
          par_intervals = (lb:dPar:ub)';
            
          % Loop over every batch:
          resampleEvery = samplingSettings{i}.resampleEvery;
          nMaxResampleGrid = ceil(nSamples / resampleEvery);
          for j=1:nMaxResampleGrid

            % Find indices for this set of numbers:
            idxMin = (j-1).*resampleEvery + 1;
            idxMax = min(j.*resampleEvery,nSamples);
              
             % Sample grid points
             parGrid = NaN(size(par_intervals,1)-1,1);
             for l=1:size(parGrid,1)
                 parGrid(l) = unifrnd(par_intervals(l),par_intervals(l+1),1,1);
             end
             
             % Sample uniform(0,1) (pseudo or quasi) random numbers:
             if use_sobol
                 Us = U(idxMin:idxMax,iCol_1);
             else
                 Us = unifrnd(0,1,idxMax - idxMin + 1,1);
             end
             
             % Sample from discretized probability distribution:
             if strcmpi(method,'unif_discrete_random_grid')
                sample_unif = lb + Us*(ub - lb);
                for k=idxMin:idxMax
                   input(k,iCol_1) = F(parGrid(find(par_intervals(1:end-1) ...
                                       <= sample_unif(k-idxMin+1),1,'last')));
                end
             elseif strcmpi(method,'trunc_norm_discrete_random_grid')
                % Compute probabilities of each observation:
                mu = samplingSettings{i}.mu;
                sig = samplingSettings{i}.sig;

                % Define CDF of truncated normal:
                pd = truncate(makedist('Normal','mu',mu,'sigma',sig),lb,ub);
                
                % Compute samples from the discretized distribution:
                sample_norm_trunc = icdf(pd,Us);
                
                for k=idxMin:idxMax
                   input(k,iCol_1) = F(parGrid(find(par_intervals(1:end-1) ...
                                   <= sample_norm_trunc(k-idxMin+1),1,'last')));
                end                
             end
             
          end
            
        else
            error(['GenNewSamples: Sampling method ''' method,...
                   ''' is not supported.']);
        end
        
        iCol_1 = iCol_2;
        
    end
    
   if reuse_numbers_for_each_H
        [~, idxSort] = sort(input(:,strcmpi(params,'H')),1);
        input = input(idxSort,:);
   elseif reuse_numbers_for_each_alpha
        [~, idxSort] = sort(input(:,strcmpi(params,'alpha')),1);
        input = input(idxSort,:);
   end
   
   if saveEvery < Inf
       idxSplit = unique([(0:saveEvery:nSamples)';nSamples]);
   else
       idxSplit = [0;nSamples];
   end
   
end

function [p, se, errs, statesOut] = EvaluateSamples(model,input,params,k,T,nWorkers,...
                                         reuse_numbers_for_each_H,...
                                         reuse_numbers_for_each_alpha,...
                                         randomSeedArray,statesIn)
% Description: Computes (call) option prices for a set of input parameters.
%
% Parameters:
%   model: [1x1 PricingModelClass] Model object.
%   input: [NxM real] N sets of input parameters of size 1xM.
%
% The remaining parameters are as described in the main function
% 'ComputeTrainingData'.
%   
% Output:
%   p:      [NxL real] Option prices (for call options). 
%   se:     [NxL real] Standard errors. 
%   errs:   [Nx1 cell] Cell array of strings containing error messages 
%           (in case some input parameters fail to execute correctly) 
%   

    nOutput = size(k,1);
    
    % Compute loop indices:
    if reuse_numbers_for_each_H
        idxH = strcmpi(params,'H');
        idxChange = find([0;diff(input(:,idxH))~=0]); 
        idxSplitLow = [1;idxChange(1:end)];
        idxSplitHigh = [idxChange(1:end)-1;size(input,1)];
        uniqH = unique(input(:,idxH));
        uniqAlpha = NaN(size(uniqH)); 
    elseif reuse_numbers_for_each_alpha
        idxAlpha = strcmpi(params,'alpha');
        idxChange = find([0;diff(input(:,idxAlpha))~=0]); 
        idxSplitLow = [1;idxChange(1:end)];
        idxSplitHigh = [idxChange(1:end)-1;size(input,1)];
        uniqAlpha = unique(input(:,idxAlpha));
        uniqH = NaN(size(uniqAlpha));
    else
        idxSplitLow = (1:size(input,1))';
        idxSplitHigh = (1:size(input,1))';
        [uniqH,uniqAlpha] = deal(NaN(1+size(idxSplitLow,1),1));
        
    end
    
    if isempty(statesIn)
        set_seed = false;
        [statesIn,randomSeedArray] = deal(cell(1+size(idxSplitLow,1),1));
    else
        set_seed = true;
    end
    
    if (reuse_numbers_for_each_H || reuse_numbers_for_each_alpha) ...
            && isfield(model.pricerSettings,'nRepeat') ...
            && model.pricerSettings.nRepeat > 1
        error(['EvaluateSamples: Parameters ''reuse_numbers_for_each_H'' ',...
               'or ''reuse_numbers_for_each_alpha'' cannot be set to true ',...
               'when model.pricerSettings.nRepeat > 1.']);
    end
    
    [p_cell, se_cell] = deal(cell(size(input,1),1));
    errs_cell = cell(size(input,1),1);
    [modelTmp,j,l,idxStart,nParamSize,exArgs,...
        p_tmp,se_tmp,errs_tmp,kTmp, Ttmp] = deal([]);
    statesOut = cell(size(statesIn));
    if set_seed;statesOut{1} = statesIn{1};end
    
    parfor (i=1:size(idxSplitLow,1),nWorkers)
    globalStream = RandStream.getGlobalStream;
        if set_seed
            RandStream.setGlobalStream(randomSeedArray{i+1});
            if ~isempty(statesIn{i+1})
                globalStream.State = statesIn{i+1};
            end
        end
        
        % Define temporary result matrices:
        [p_tmp, se_tmp] = deal(NaN(nOutput,idxSplitHigh(i)-idxSplitLow(i)+1));
        errs_tmp = cell(idxSplitHigh(i)-idxSplitLow(i)+1,1);
        try
            % Copy model object:
            modelTmp = model.DeepCopy();
            if reuse_numbers_for_each_H
                %Pre-generate Monte Carlo stuff
                modelTmp.H = uniqH(i);
                modelTmp.GenNumbersForMC(T,'Y_and_dW1');
                reuse_numbers = true;
            elseif reuse_numbers_for_each_alpha
                %Pre-generate Monte Carlo stuff
                modelTmp.alpha = uniqAlpha(i);
                modelTmp.GenNumbersForMC(T,'Y');
                reuse_numbers = true;
            else
                reuse_numbers = false;
            end

            % Loop over samples allocated to this thread:
            idx = 0;
            for j=idxSplitLow(i):idxSplitHigh(i)
                idx = idx + 1;
                tic;
                try
                    % Modify parameters:
                    idxStart = 0;
                    for l=1:size(params,2)
                        idxStart = idxStart + 1;
                        if isa(model.(params{l}),'CurveClass')
                            nParamSize = size(model.(params{l}).values,1);
                            modelTmp.(params{l}).values = ...
                                        input(j,idxStart:idxStart+nParamSize-1)';
                            idxStart = idxStart + nParamSize - 1;
                        else
                            modelTmp.(params{l}) = input(j,idxStart);
                        end
                    end

                    % Get prices:
                    if isa(modelTmp.pricerSettings,'MonteCarloPricerSettingsClass')
                        [p_tmp(:,idx),~,~,~,se_tmp(:,idx)] = modelTmp.GetPrices(...
                            k,T,false,'priceType','price',...
                            'standardErrors',true,'optionType','call',...
                            'use_existing_numbers',reuse_numbers);
                    else
                        p_tmp(:,idx) = modelTmp.GetPrices(k,T,false,...
                            'priceType','price','optionType','call',...
                            'use_existing_numbers',reuse_numbers);
                    end
                
               catch e
                    % Store error message in table
                   errs_tmp(idx) = {e.message};
               end 
                sec = toc;
                disp(['Iteration ', num2str(j), ' completed in ', ...
                    num2str(sec), ' seconds']);

            end
        catch
            
        end
        p_cell{i} = p_tmp;
        se_cell{i} = se_tmp;
        errs_cell{i} = errs_tmp;
        
        if set_seed
            tmp = rng;
            statesOut{i+1} = tmp.State;
        end
        
    end
    
    if ~set_seed
        statesOut = [];
    end
    
    % Unpack results:
    [p, se] = deal(NaN(size(input,1),size(k,1)));
    errs = cell(size(errs_cell));
    for i=1:size(idxSplitLow,1)
        p(idxSplitLow(i):idxSplitHigh(i),:) = p_cell{i}';
        se(idxSplitLow(i):idxSplitHigh(i),:) = se_cell{i}';
        errs(idxSplitLow(i):idxSplitHigh(i)) = errs_cell{i};
    end
    
end






