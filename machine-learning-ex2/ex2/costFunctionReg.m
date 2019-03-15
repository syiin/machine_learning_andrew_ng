function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

thetaTX = X * theta
g = sigmoid(thetaTX)
yIsOne = (-1 .* y)' * log(g)
yIsZero = (1 .- y)' * log(1 .- g) 
initialJ = (yIsOne - yIsZero) / m

thetaOneUp = theta(2:end, :)
thetaSQ = thetaOneUp' * thetaOneUp
regVal = (lambda / (2 * m)) * thetaSQ

J = initialJ + regVal

initialGrad = (X' * (g .- y) ) / m
regExp = (lambda / m) * thetaOneUp
rowOnes = zeros(1, (size(regExp)(2)))
regExp = [rowOnes; regExp]

grad = initialGrad + regExp




% =============================================================

end
