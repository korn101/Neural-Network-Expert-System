nAlpha = [ 0.01 0.02 0.03 0.04 0.05 ];
epochsToBestAcc = [ 40 13 6 8 12 ];

plotFig = figure;

axis auto;
plot(nAlpha, epochsToBestAcc, 'r');
xlabel('Alpha Value');
ylabel('Epochs to best accuracy');
title('Plot of epochs to reach best accuracy vs Alpha Values');


print(plotFig, 'DataPlot2', '-dpng');