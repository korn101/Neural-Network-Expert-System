classdef NeuralNetwork < matlab.mixin.Copyable
    properties (SetAccess = private, GetAccess = public)
        
        % learning rate
        alpha = 0.01;
        
        noInputs;
        noHiddenNeurons;
        noOutputNeurons;
        
        constBias = -1;
        
        neuralLayers = {[]}; % store hidden and output layer.
        
        % backpropagation calculations:
        arrSquareErrors; % array of square errors.
        % arrNeuronDelta; % delta value for neuron. MOVED AS NEURONLAYER.
       
    end
    
    methods
        
        function LR = GetLearnRate(NEURALNETWORK)
            LR = NEURALNETWORK.alpha; 
        end
        
        function SetLearnRate(NEURALNETWORK, newA)
            NEURALNETWORK.alpha = newA;
        end
        
        function NEURALNETWORK = NeuralNetwork(inputs, hidden, outputs)
            % set some properties
            NEURALNETWORK.noInputs = inputs;
            NEURALNETWORK.noHiddenNeurons = hidden;
            NEURALNETWORK.noOutputNeurons = outputs;
            % create the layers:
            
            NEURALNETWORK.neuralLayers{1} = NeuronLayer(hidden, inputs);
            NEURALNETWORK.neuralLayers{2} = NeuronLayer(outputs, hidden);
            
            % read the file
            
            
            
        end
        
        
        function debugLayer(NEURALNETWORK, layerNo)
            % debug the layerNo.
            
            for i=1:NEURALNETWORK.neuralLayers{layerNo}.noOfNeurons
                fprintf('Neuron:%d of layer %d :-----\n', i, layerNo);
                NEURALNETWORK.neuralLayers{layerNo}.arrNeurons{i}.printDebug;
            end
            
        end
        
        
        function Update(NEURALNETWORK, inputArr)
            % take an input array,
            % pass it through,
            % present output results.
            
            % check that we have been passed an array:
            if (size(inputArr) - NEURALNETWORK.noOutputNeurons ~= NEURALNETWORK.noInputs)
                fprintf('Neural Net Inputs out of range\n');
            else
                
                % for each layer of the neuralLayers:
                for iLayer = 1:size(NEURALNETWORK.neuralLayers,2)
                    % for each NEURON:
                    for j=1:NEURALNETWORK.neuralLayers{iLayer}.noOfNeurons
                        % sum the inputs*weights for each neuron.
                        sumInputs = 0;
                        
                        % now for each weight/input:
                        % if an input layer then add with the inputs, if
                        % NOT then we need to get the outputs from the
                        % previous layer as our new inputs.
                        if iLayer == 1
                            % if we are proccessing the first layer.
                            for k=1:NEURALNETWORK.neuralLayers{iLayer}.arrNeurons{j}.noInputs
                                sumInputs = sumInputs + (NEURALNETWORK.neuralLayers{iLayer}.arrNeurons{j}.arrWeights(k) .* inputArr(k));
                            end
                            
                        else
                        
                            for k=1:NEURALNETWORK.neuralLayers{iLayer}.arrNeurons{j}.noInputs
                                sumInputs = sumInputs + (NEURALNETWORK.neuralLayers{iLayer}.arrNeurons{j}.arrWeights(k) .* NEURALNETWORK.neuralLayers{iLayer-1}.arrNeurons{k}.sigmoidOut);
                            end

                        end
                        
                        % then add the bias.
                        
                        sumInputs = sumInputs + (NEURALNETWORK.neuralLayers{iLayer}.arrNeurons{j}.arrWeights(NEURALNETWORK.neuralLayers{iLayer}.noOfInputs + 1) .* NEURALNETWORK.constBias);
                        
                        % now store the output of each neuron, after
                        % passing it through the sigmoid.
                        
                        NEURALNETWORK.neuralLayers{iLayer}.arrNeurons{j}.sigmoidPre = sumInputs;
                        
                        NEURALNETWORK.neuralLayers{iLayer}.arrNeurons{j}.sigmoidOut = Sigmoid(sumInputs);
                        
                    end
                    
                end % for each layer
                
            end
            
            
        end
        
        % back propagate the neural network.
        function BackPropagate(NEURALNETWORK, recordLine)
            
            % where recordLine
            % 9 inputs, yes, no
            
            desiredOutputs = zeros(2);
            dataInputs = zeros(NEURALNETWORK.noInputs);
            
            % read the record line and extract the desiredOutputs outputs.
            for i=1:2
                desiredOutputs(i) = recordLine(NEURALNETWORK.noInputs + i);
            end
            
            for i=1:NEURALNETWORK.noInputs
                dataInputs(i) = recordLine(i);
            end
            
            
            % calculate square error (not mean square)
            % get delta's of output layer (delta21, delta22)
            % then get deltas (neuron weights page):
            % calculate associated neuron error E
            % then calculate delta11
            % use delta to calculate the gradient.
            % del = -2*delta*Xk ( Xk = input )
            % then our new weights are:
            % weight <- weight + u*(-del(k))
            % = weight + 2*u*delta(k)*Xk
            % (where u is learning rate)
            
            
            
            % error for ONLY the output layer. If not output layer then use
            % alternative.
            
            % REMOVED
            
            %for i=1:NEURALNETWORK.neuralLayers{2}.noOfNeurons
            %    NEURALNETWORK.arrSquareErrors(i) = (desiredOutputs(i) - NEURALNETWORK.neuralLayers{2}.arrNeurons{i}.sigmoidOut).^2;
            %end
            
            % TODO:
            % make the remainder of this code work for the rest of the
            % layers. Every layer up till the input. Remeber to correctly
            % implement the calculation of errors.
            
            for layerIteration=2:-1:1
                
                % debug fprintf('ENTERING ITERATION %d\n', layerIteration);
                
                % delta = error*activate'(sigmoidPre)
                % calculate neuron deltas for noNeurons in layer
                
                for i=1:NEURALNETWORK.neuralLayers{layerIteration}.noOfNeurons
                    
                    % can only use this form for output layer neurons.
                    % in the case of others, error must be calculated
                    % differently. More precisely:
                    % using: E = sum( weightToNextLayer * deltaOfNextLayer )
                    % for all neurons that this one connects to. Since this
                    % neural network is homogenous we can assume that its the
                    % sum for the total.
                    
                    
                    if layerIteration == size(NEURALNETWORK.neuralLayers, 2)
                        % if we are doing the output layer, then its just the
                        % desiredOutputs - actual = E.
                        NEURALNETWORK.neuralLayers{layerIteration}.arrNeuronDelta(i) = (desiredOutputs(i) - NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.sigmoidOut) .* SigmoidDerivative( NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.sigmoidPre );
                    else
                        % else, E must be specially calculated.
                        NEURALNETWORK.neuralLayers{layerIteration}.arrNeuronDelta(i) = (NEURALNETWORK.getNeuronError(layerIteration, i)) .* SigmoidDerivative( NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.sigmoidPre );
                    end
                end
                
                % now we can update the weights of the current layers neurons.
                
                for i=1:NEURALNETWORK.neuralLayers{layerIteration}.noOfNeurons
                    % every neuron of this layer i
                    for k=1:NEURALNETWORK.neuralLayers{layerIteration}.noOfInputs
                        % every weight k of neuron i
                        
                        % for every weight: noWeights = noOfInputs + 1, (for
                        % the bias)
                        
                        % if this is the terminal layer on input side then
                        % we cant get a sigmoid out from the previous
                        % layer. Instead we use the X input from the
                        % dataset.
                        if layerIteration == 1
                            % using x input from inputs:
                            
                            NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.UpdateWeight(k, NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.arrWeights(k) + (2.*NEURALNETWORK.alpha.*NEURALNETWORK.neuralLayers{layerIteration}.arrNeuronDelta(i).*dataInputs(k)));
                            
                        else
                        
                            NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.UpdateWeight(k, NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.arrWeights(k) + (2.*NEURALNETWORK.alpha.*NEURALNETWORK.neuralLayers{layerIteration}.arrNeuronDelta(i).*NEURALNETWORK.neuralLayers{layerIteration - 1}.arrNeurons{k}.sigmoidOut));
                        
                        %NEURALNETWORK.neuralLayers{2}.arrNeurons{i}.arrWeights(k) = NEURALNETWORK.neuralLayers{2}.arrNeurons{i}.arrWeights(k) + 2*NEURALNETWORK.alpha*NEURALNETWORK.arrNeuronDelta(i)*NEURALNETWORK.neuralLayers{2 - 1}.arrNeurons{k}.sigmoidOut;
                        % NOTE: the use of the previous layer.
                        end
                        
                    end
                    
                    % for the bias weight:
                    NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.UpdateWeight(NEURALNETWORK.neuralLayers{layerIteration}.noOfInputs+1, NEURALNETWORK.neuralLayers{layerIteration}.arrNeurons{i}.arrWeights(NEURALNETWORK.neuralLayers{layerIteration}.noOfInputs+1) + 2*NEURALNETWORK.alpha*NEURALNETWORK.neuralLayers{layerIteration}.arrNeuronDelta(i)*NEURALNETWORK.constBias);
                    
                end
                
            end
            
        end
        
        function [ SERRORS1, SERRORS2 ] = getSquareErrors(THIS, sRecord)
            
            THIS.Update(sRecord); % update feed forward.
   
            desiredOutputs = zeros(2);
            
            % read the record line and extract the desiredOutputs outputs.
            for i=1:2
                desiredOutputs(i) = sRecord(THIS.noInputs + i);
            end
            
            
            for i=1:THIS.neuralLayers{2}.noOfNeurons
                THIS.arrSquareErrors(i) = (desiredOutputs(i) - THIS.neuralLayers{2}.arrNeurons{i}.sigmoidOut).^2;
            end
            
            SERRORS1 = THIS.arrSquareErrors(1);
            SERRORS2 = THIS.arrSquareErrors(2);
            
        end
        
        % this utility function gets the associated E value of a neuron of
        % a specific layer.
        % in other words, it finds the associated weights (not including
        % bias) and gets the sum of them as well as the associated deltas
        % with the layer.
        function E=getNeuronError(THIS, layerNo, neuronNo)
            
            result = 0;
            
            
            for i=1:THIS.neuralLayers{layerNo + 1}.noOfNeurons
                % for i every neuron in the next layer.
                result = result + (THIS.neuralLayers{layerNo + 1}.arrNeurons{i}.arrWeights(neuronNo) .* THIS.neuralLayers{layerNo + 1}.arrNeuronDelta(i));
            end
            
            E=result;
            
        end
        
        function one = Process(THIS, inputAr )
        
            THIS.Update(inputAr); % feed forward.
            one = THIS.neuralLayers{2}.arrNeurons{1}.sigmoidOut;
            
            fprintf('Output of Neuron 1: %f \nRounded Output: %d\n', one, round(one));
            
        end
        
        
    end
end

% inputs -> hiddenLayer --> outputLayer

