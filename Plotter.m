epochBestTest = [ 17 13 12 48 43 44 59 68 ];
accuracy = [ 95 95 95 98 98 98 98 98 ];
noHiddens = [ 2 3 4 5 6 7 8 9 ];
bestTestError = [ 9.67 7.76 6.99 4.73 4.8 4.77 4.63 4.60 ];

plotFig = figure;

axis auto;
subplot(3,1,1);
plot(noHiddens, epochBestTest, 'r');
xlabel('Number of hidden neurons');
ylabel('Epoch till best test error');
title('Plot of epochs needed to get best test error for different nHidden');

subplot(3,1,2);
plot(noHiddens, accuracy, 'b');
xlabel('Number of hidden neurons');
ylabel('Prediction Accuracy (%%)');
title('Plot of predictive test accuracy vs nHidden');

axis auto;
subplot(3,1,3);
plot(noHiddens, bestTestError, 'r');
xlabel('Number of hidden neurons');
ylabel('Best Test Errors in Percent');
title('Plot of lowest capable test error value vs nHidden');

print(plotFig, 'DataPlot1', '-dpng');