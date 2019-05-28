function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

options = [0.01 0.03 0.1 0.3 1 3 10 30];
options = options(:);
optionsLength = size(options, 1);
predLength = size(options, 1).^2;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

allPredictions = zeros(optionsLength, 3);
currentC = options(1, 1);
currentSigma = options(1, 1);
for i=1:optionsLength
    currentC = options(i, 1);
    for j=1:optionsLength    
        currentSigma = options(j, 1);
        model = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma)); 
        predictions = svmPredict(model, Xval);
        currentPredError = mean(double(predictions ~= yval));
        allPredictions(j + ((i-1)*10), :) = [currentC currentSigma currentPredError]; 
    end
end

disp("all predictions w zeros");
disp(allPredictions);
allPredictions = nonzeros(allPredictions);
allPredictions = reshape(allPredictions, predLength, 3);
disp("all predictions wo zeros");
disp(allPredictions);

[minVakl, minIdx] = min(allPredictions(:, 3));
disp(minIdx);
disp(allPredictions(minIdx, :));
C = allPredictions(minIdx, 1);
sigma = allPredictions(minIdx, 2);




% =========================================================================

end
