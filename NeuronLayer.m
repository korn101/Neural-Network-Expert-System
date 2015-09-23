classdef NeuronLayer < matlab.mixin.Copyable
    properties (SetAccess = public, GetAccess = public)
        
        noOfNeurons; % number of neurons in this layer.
        arrNeurons = cell(1); % the array / cell list of neurons.
        noOfInputs; % number of inputs to the neurons in this layer.
        
        arrNeuronDelta; % delta value for each neuron in this layer.
        
    end
    
    methods
        
        function NEURONLAYER = NeuronLayer(noNeurons, noInputsPer)
            
            % initialize some random neurons.
            for i = 1:noNeurons
                
                NEURONLAYER.arrNeurons{i} = Neuron(noInputsPer);
                
            end
            
            NEURONLAYER.noOfNeurons = noNeurons;
            NEURONLAYER.noOfInputs = noInputsPer;
            NEURONLAYER.arrNeuronDelta = zeros(noNeurons);
            
        end
      
        
    end
end