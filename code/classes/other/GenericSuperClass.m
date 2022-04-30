classdef GenericSuperClass < handle
% Description: This class contains some useful generic class features.

methods
   function obj = GenericSuperClass()
   % Description: Constructor.
   %
   % Output:
   %   [1x1 GenericSuperClass] The object.
   % 
       
   end 
   function ParseConstructorInputs(obj,varargin)
   % Description: Takes name-value pair inputs and parses them to 
   % the subclass property members. Objects are copied using the 'DeepCopy'
   % if it is implemented. Otherwise we use the assignment operator. 
   %
   % Parameters:
   %    varargin: [1xN cell] Sequence alternating between strings referring
   %              to object properties and the values or objects that should be
   %              assigned to them. Thus N must be a multiple of 2.
   %

        nvarargin = size(varargin,2);
        errMsg = ['Object construction failed: Inputs must be in ', ...
                  'name-value pairs.'];
        if ~mod(nvarargin,2)==0
            error(errMsg);
        end

        allProps = properties(obj);
        for i=1:(nvarargin/2)
            nm = varargin{2*i-1};
            val = varargin{2*i};

            if ~ischar(nm);error(errMsg);end

            % Check if property even exists:
            idxPropMatch = strcmp(properties(obj),nm);
            if ~any(idxPropMatch)
                error(['Object construction failed: Property ', nm, ...
                       ' does not exist.']);
            end
            prop = allProps(idxPropMatch);
            prop = prop{1};

           % Copy value:
            dc = any(strcmp(methods(val),'DeepCopy'));
            if dc;obj.(prop) = val.DeepCopy();else;obj.(prop) = val;end

        end           

   end
   function newObj = DeepCopy(obj)
   % Description: Attempts to perform a deep copy of the object. See bullet
   % point (2) below for when this is actually a true deep copy.
   %
   % Remarks: 
   %    (1) The class must allow the creation of an empty object.
   %    (2) The code loops over all properties in the object and attempts to 
   %        call the 'DeepCopy' method on each property member. If a property
   %        member does not have the a 'DeepCopy' method we use the default
   %        assignment operator '=' which may or may not perform a deep
   %        copy.
   %
   % Output:
   %    newObj: [1x1 GenericSuperClass] A copy of the object.
   %
   % Example: newObj = obj.DeepCopy()
   %

        % Create new empty object:
        construct = str2func(class(obj));
        newObj = construct();

        % Loop over properties:
        props = properties(obj);
        for iprop = 1:length(props)
          thisprop = props{iprop};
          % Use 'DeepCopy' if possible:
          mAvail = methods(obj.(thisprop));
          if any(strcmp(mAvail,'DeepCopy'))
              newObj.(thisprop) = obj.(thisprop).DeepCopy;
          else
              newObj.(thisprop) = obj.(thisprop);
          end
        end

   end
end

end

