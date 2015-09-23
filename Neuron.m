classdef Neuron < matlab.mixin.Copyable
    properties (SetAccess = private, GetAccess = public)
        noInputs; % number of inputs into neuron
        arrWeights; % weights for each input
    end
    
    properties (SetAccess = public)
        sigmoidOut; % output after sigmoid.
        sigmoidPre; % pre sigmoid output.
    end
    
    methods
        
        function NEURON = Neuron(noinputs)
            NEURON.noInputs = noinputs;
            
            % initialize weights:
            for i=1:NEURON.noInputs+1 % include the BIAS weight.
                % for good initialisation, 
                % initialize the random number between -1/sqrt(
                NEURON.arrWeights(i) = clampedRandom();
            end
            
            
        end
        
        function UpdateWeight(NEURON, wNo, val)
           
            NEURON.arrWeights(wNo) = val;
            %NEURON.arrWeights(wNo) = 0;
            
        end
        
        function printDebug(NEURON)
           
            fprintf('s:%f x:%f\n', NEURON.sigmoidPre, NEURON.sigmoidOut);
            
            for i=1:NEURON.noInputs+1
                
                fprintf('w:%f', NEURON.arrWeights(i)); 
                if (i==NEURON.noInputs+1)
                    fprintf(' (bias)\n');
                else
                    fprintf('\n');
                end
            end
        end
        
        
        
        
    end
    
end