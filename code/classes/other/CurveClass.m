classdef CurveClass < GenericSuperClass
% Description: Defines a general curve object. A curve is here defined by 
% specifying a set of fixed points (t_i,y_i), i=1,...,N, and then placing
% an interpolation and extrapolation method on top.
%
% Properties:
%       o gridpoints:    [Nx1 real] Grid points t_i, i = 1,...,N. Must be sorted 
%                        in ascending order.
%       o values:        [Nx1 real] Curve values y_i, i = 1,...,N.
%       o interpolation: [1x1 string] Interpolation method, options are 'flat' 
%                        and 'linear'.
%       o extrapolation: [1x1 string] Extrapolation method, currently the only 
%                        option is 'flat.
%

properties
    gridpoints    
    values        
    interpolation 
    extrapolation 
end
    
methods
    function obj = CurveClass(varargin)
    % Description: Constructor.
    %
    % Parameters:
    %   varargin: Inputs must come in name-value pairs corresponding to
    %   each property of the object. The properties of the object will
    %   then be set to the values. Default values may be chosen if some
    %   properties are not set.
    %
    % Output:
    %   [1x1 CurveClass] The object.
    %
    % Example: CurveClass('gridpoints',[0.1;1],'values',[0.02;0.04])
    %

        % Parse input name-value pairs to object properties:
        obj.ParseConstructorInputs(varargin{:});

        % Validation checks:
        if ~isequal(size(obj.values),size(obj.gridpoints)) ...
                || size(obj.values,2) > 1 || size(obj.gridpoints,2) > 1
            error(['CurveClass: Grid points and values',...
                   ' must be column vectors of the same size.']);
        end

        % Default values:
        if isempty(obj.interpolation)
            obj.interpolation = 'flat';
        else
            if ~(strcmpi(obj.interpolation,'flat') || ...
                 strcmpi(obj.interpolation,'linear'))
              error('CurveClass: Invalid choice of interpolation method.');
            end
        end
        if isempty(obj.extrapolation)
            obj.extrapolation = 'flat';
        else
            if ~strcmpi(obj.extrapolation,'flat')
              error('CurveClass: Invalid choice of extrapolation method.');
            end        
        end

    end
    function vals = Eval(obj,pts)
    % Description: Evaluates the curve using interpolation and extrapolation 
    % as necessary.
    %
    % Parameters:
    %   pts: [Nx1 real] Evaluation points.
    %
    % Output:
    %   vals: [Nx1 real] Value of curve at evaluation points.
    %
    % Example: obj.Eval((0.1:0.1:1)')
    %

        if ~strcmpi(obj.extrapolation,'flat')
            error(['CurveClass:GetValues: Extrapolation method', ...
                  ' is not supported.']);
        end

        switch obj.interpolation
            case 'linear'
                [maxPt, idxMax] = max(obj.gridpoints);
                [minPt, idxMin] = min(obj.gridpoints);
                idxHigh = pts >= maxPt;
                idxLow = pts <= minPt;
                idxMiddle = ~idxHigh&~idxLow;
                vals = NaN(size(pts));
                vals(idxMiddle) = interp1(obj.gridpoints,obj.values,pts(idxMiddle));
                if any(idxHigh)
                    vals(idxHigh) = obj.values(idxMax);
                end
                if any(idxLow)
                    vals(idxLow) = obj.values(idxMin);
                end                
            case 'flat'
                n = size(obj.gridpoints,1);
                vals = reshape(obj.values(min(...
                                sum(bsxfun(@(a,b)(a < b) ,...
                                obj.gridpoints,pts(:)'), 1) + 1, n)), ...
                                size(pts, 1), size(pts, 2));                            
            otherwise
                error(['CurveClass:GetValues: Interpolation method', ...
                      ' is not supported.']);
        end

    end
    function vals = Integrate(obj,a,b)
    % Description: Returns the integrated curve.
    %
    % Parameters:
    %   a: [Nx1 real] Lower end-points.
    %   b: [Nx1 real] Upper end-points.
    % 
    % Output:
    %   vals: [Nx1 real] Curve integrated from a to b.
    %
    
        if size(a,1) ~= size(b,1)
            error('CurveClass:Integrate: The input vectors must have the same length');
        end

        vals = NaN(size(a));
        for i=1:size(a,1)
            vals(i) = integral(@(t)(obj.Eval(t)),a(i),b(i));
        end
    
    end
end

end

