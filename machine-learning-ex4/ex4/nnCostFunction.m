function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% rowOnes = zeros(1, (size(regExp)(2)))
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% Add column of ones to X
X = [ones(rows(X), 1) X];
% Encode y to 0 & 1
y_matrix = eye(num_labels)(y,:) 
% returns 5000 x 25, 25 nodes for each sample of x
z2 = X * Theta1' 
a2 = sigmoid(z2)
% returns 5000 x 10, each row a value of y
a2 = [ones(rows(a2), 1) a2];
z3 = a2  * Theta2'
a3 = sigmoid(z3)

yIsOne = (-1 .* y_matrix) .* log(a3)
yIsZero = (1 .- y_matrix) .* log(1 .- a3)
difference = yIsOne - yIsZero 
initialJ = sum(difference(:)) / m

% Remove Theta0
Theta1_no0 = Theta1(:, 2:end)
Theta2_no0 = Theta2(:, 2:end)
reg1 =  Theta1_no0 .^ 2
reg1 = sum(reg1(:))
reg2 = Theta2_no0 .^ 2
reg2 = sum(reg2(:))
regExp = (reg1 + reg2) * (lambda / (2 * m))
J = initialJ + regExp



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Returns a 5000 x 10 matrix with the differnce between prediction and y

delta_3 =  a3 - y_matrix
delta_2 = (delta_3 * Theta2_no0) .* sigmoidGradient(z2)

Theta1_grad = (delta_2' * X) / m
Theta2_grad = (delta_3' * a2) / m

% delta_2 = (delta_3 * Theta2_no0) .* sigmoidGradient(z2)
% delta_2 = delta_2(2:end)
% Theta1_grad = delta_2 * a1' / m




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



gradReg1 = (lambda / m) * Theta1_no0
gradReg1 = [zeros(rows(gradReg1), 1) gradReg1]
Theta1_grad = Theta1_grad + gradReg1

gradReg2 = (lambda / m) * Theta2_no0
gradReg2 = [zeros(rows(gradReg2), 1) gradReg2]
Theta2_grad = Theta2_grad + gradReg2


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
