% Train.m
% This class is used to train the network.

% first, clear all junk.

close all;
clear;
clc;

% the fundamental principle of the training is to run a feed-forward update
% and then apply backpropagation. (for each line of sample).
% A total iteration through all ~700 total records = 1 epoch.

rawData = load('rawdata.dt'); % first load the dataset:
maxRecords = size(rawData,1); % we know max number of records in the data is:

trainLimit = 350;
validationLimit=175;
testLimit=174;

% section the different data sets. OR pick randomly? probably better.
%trainingSet = rawData(1:trainLimit, 1:11);
%validationSet = rawData(trainLimit+1:trainLimit+validationLimit, 1:11);
%testSet = rawData(trainLimit+validationLimit+1:trainLimit+validationLimit+testLimit, 1:11);

% test the effect of learning rate.

    
for nHidNeuron = 6:6 % for no hidden neurons.

    acc = figure;
    SetGlobalHidden(nHidNeuron);
    
    maxEpochs = 30; % max Epochs to run.

    epochError = zeros(1, maxEpochs);
    avgTestScoreAtEpoch = zeros(1, maxEpochs);
    accuracyAtEpoch = zeros(1, maxEpochs);
    
    validationError = zeros(1, maxEpochs);

    initialError = 0;

    NN1 = NeuralNetwork(9,nHidNeuron,2); % create network.
    NN1.SetLearnRate(0.02);

    fprintf('== Initial Network ==\n');
    NN1.debugLayer(2);

    % before we run lets start timer.
    tTicker = tic;

    boolValidationBelowThreshold = false;
    
    if nHidNeuron > 4
        validationThreshold = 0.02; % only detect minima once we go lower than this.
    else
        validationThreshold = 0.07; %0.9;
    end
    
    
    for nEpoch = 0:maxEpochs

        sumE = 0;
        sumV = 0;
        randomRecord = 0;
        randomValidation = 0;

        for nRecord = 1:trainLimit

            %randomRecord = rawData(round(RangedRandom(1,trainLimit)),1:11);

            randomRecord = rawData(nRecord,1:11);
            
            if nEpoch > 0           
                % for every record feed forward and back-prop.
                NN1.Update(randomRecord); % forward update
                NN1.BackPropagate(randomRecord); % back-prop
            end


            [ sqrs1, sqrs2 ] = NN1.getSquareErrors(randomRecord);
            % calc square errors after backprop.
            % instead lets get the average amount of square error per epoch
            % sum the errors.
            %sqrE1(nEpoch, nRecord) = sqrs1;
            %sqrE2(nEpoch, nRecord) = sqrs1;

            sumE = sumE + ( sqrs1 + sqrs2 );

        end

        % after each epoch, we must also calculate the error for the validation
        % samples.

        for nValidationRecord = 1:validationLimit

            %randomValidation = rawData(round(RangedRandom(trainLimit+1,trainLimit+validationLimit)),1:11);
            randomValidation = rawData(trainLimit+nValidationRecord,1:11);

            % now test the record on the training thus far:
            % return the square errors on our validation sample:
            [ sqrsV1 , sqrsV2 ] = NN1.getSquareErrors(randomValidation);

            sumV = sumV + (sqrsV1 + sqrsV2);

        end


        if nEpoch > 0
            epochError(nEpoch) = sumE ./ trainLimit;
            validationError(nEpoch) = sumV ./ validationLimit;
        else
            initialError = sumE ./ trainLimit; % possibly not needed.
            if ((sumV ./ validationLimit) < validationThreshold)
                validationThreshold = sumV ./ validationLimit;
            end
        end


        fprintf('== After Epoch %d ==\n', nEpoch);
        NN1.debugLayer(2);

        xAxis = 1:nEpoch;

        %axis auto;

        % medfilt1
        
        % real time plot
        %subplot(2,1,1);
        %plot(xAxis, (epochError(1:nEpoch)), 'b', xAxis, (validationError(1:nEpoch)), 'r');
        %grid on;
        %xlabel('Epoch Number');
        %ylabel('Sum of Square Output errors');
        %title('Plot of learning characteristics (smoothing filter)');
        %legend('Training Error', 'Validation Error');
        %drawnow;

        % it has been determined that the terminal condition for epoch count:
        % if the validationError drops below 0.05 and then proceeds to rise
        % above it once again we can exit the epoch loop.

        finalEpoch = nEpoch;
        
        % test for minimums
        
        if nEpoch > 0
            if validationError(nEpoch) < validationThreshold 
                boolValidationBelowThreshold = true;
                validationThreshold = min(validationError(1:nEpoch));
            else
                if boolValidationBelowThreshold == true
                    % if this is satisfied then we can break.
                    finalEpoch = nEpoch;
                    break; % break from current nEpoch.
                end
                
                %validationThreshold = min(validationError(1:nEpoch));
            end
            
        end

        % after training to epoch nEpoch.
        
        % after testing the epochs, lets look at test cases.
        
        testScores = zeros(1);
        randomTest = 0;
        failedTests = 0;
        
        for nTestRecord=1:testLimit
            % for each test record
            %randomTest = rawData(round(RangedRandom(trainLimit+validationLimit+1,trainLimit+validationLimit+testLimit)),1:11);
            
            % aternatively test every single testData:
            randomTest = rawData(trainLimit+validationLimit+nTestRecord,1:11);

            % now test the record on the training thus far:
            % return the square errors on our validation sample:
            [ sqrsT1 , sqrsT2 ] = NN1.getSquareErrors(randomTest);
            
            testScores(nTestRecord) = (sqrsT1 + sqrsT2);
            if testScores(nTestRecord) >= 0.5
                failedTests = failedTests + 1;
            end
        end
        
        if nEpoch > 0
            accuracyAtEpoch(nEpoch) = round((1 - failedTests./testLimit).*100);
        end
        % test score average = 
    
        if nEpoch > 0
            avgTestScoreAtEpoch(nEpoch) = sum(testScores) ./ testLimit;
        end
        
        % real time plot
        subplot(2,1,1);
        plot(xAxis, (epochError(1:nEpoch)), 'b', xAxis, (validationError(1:nEpoch)), 'r', xAxis, avgTestScoreAtEpoch(1:nEpoch), '-g');
        grid on;
        xlabel('Epoch Number');
        ylabel('Sum of Square Output errors');
        [ tbest , itbest ] = min(avgTestScoreAtEpoch(1:nEpoch));
        title(sprintf('Plot of learning characteristics for %d hidden neurons at alpha=%0.3f.\nBest Test Error is: %f at Epoch %d',NN1.noHiddenNeurons,NN1.GetLearnRate(), tbest, itbest));
        legend('Training Error', 'Validation Error', 'Test Error');
        %drawnow;
        
        
        averageAccuracy(1:nEpoch) = mean(accuracyAtEpoch(1:nEpoch));
        subplot(2,1,2);
        plot(xAxis, accuracyAtEpoch(1:nEpoch), 'r', xAxis, averageAccuracy(1:nEpoch), 'g');
        grid on;
        xlabel('Epoch Number');
        ylabel('Test Prediction Accuracy (%%)');
        [ xbest, ibest ] = max(accuracyAtEpoch(1:nEpoch));
        title(sprintf('Plot of test set prediction accuracy\nBest accuracy is %d%% at epoch %d', xbest, ibest));
        text(ibest, xbest, sprintf('Max: %d%% -> ', xbest), 'HorizontalAlignment','right');
        drawnow;
        
    end
    

    % after the train is done, save the elapsed time.
    fprintf('===\nTraining of %d epochs completed in ~ %d minutes\n===\n', finalEpoch, round(toc(tTicker) ./ 60));

    % after testing the epochs, lets look at test cases.

    % since test scores is not done in every iteration:
    
    % comment out.
%     testScores = zeros(1);
%     randomTest = 0;
%     failedTests = 0;
% 
%     for nTestRecord=1:testLimit
%         % for each test record
%             randomTest = rawData(round(RangedRandom(trainLimit+validationLimit+1,trainLimit+validationLimit+testLimit)),1:11);
% 
%             % now test the record on the training thus far:
%             % return the square errors on our validation sample:
%             [ sqrsT1 , sqrsT2 ] = NN1.getSquareErrors(randomTest);
% 
%             testScores(nTestRecord) = (sqrsT1 + sqrsT2);
%             if testScores(nTestRecord) > 0.5
%                 failedTests = failedTests + 1;
%             end
%     end
% 
%     % anything above 1 is bad.
% 
%     acc = figure;
%     subplot(2,1,1);
%     plot(xAxis, smooth(epochError(1:nEpoch)), 'b', xAxis, (validationError(1:nEpoch)), 'r');
%     grid on;
%     xlabel('Epoch Number');
%     ylabel('Sum of Square Output errors');
%     title('Plot of learning characteristics (smoothing filter)');
%     legend('Training Error', 'Validation Error');
%     drawnow;
% 
%     
%     accuracy = round((1 - failedTests./testLimit).*100);
% 
%     % test score average = 
%     
%     testScoreAverage = sum(testScores) ./ testLimit;
%     
%     subplot(2,1,2);
%     plot(testScores);
%     grid on;
%     xlabel('Test Number');
%     ylabel('Sum of Square Output errors');
%     title(sprintf('Plot of test results for %d hidden neurons. \nAverage Sum of Square Test Errors = %.3f\nTests < 0.5 = %d %%', NN1.noHiddenNeurons, testScoreAverage, accuracy));
%     drawnow;
     print(acc, sprintf('%d neurons - a %.03f - time %d mins.png', NN1.noHiddenNeurons, NN1.GetLearnRate(), round(toc(tTicker) ./ 60) ), '-dpng');
% 
     save(sprintf('data-lastrun-alpha-%.03f-hiddens-%d.dat', NN1.GetLearnRate(), NN1.noHiddenNeurons));
%     acc=figure('visible','off');
    
end