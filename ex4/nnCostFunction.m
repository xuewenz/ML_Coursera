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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y2 = zeros(m,num_labels);

for i = 1:length(y)
    y2(i,y(i)) = 1;
end

X = [ones(m,1) X];

a2 = sigmoid(X*Theta1');
a2 = [ones(m,1) a2];
a3 = sigmoid(a2*Theta2');

Theta1sq = Theta1.^2;
Theta2sq = Theta2.^2;
Theta1sq(:,1) = 0;
Theta2sq(:,1) = 0;

J_unReg = (1/m)*sum(-y2.*log(a3) - (1-y2).*log(1-a3), 'all');
Reg = (lambda/(2*m))*(sum(Theta1sq,'all') + sum(Theta2sq,'all'));
J = J_unReg + Reg;

delta2T = 0;
delta1T = 0;

for t = 1:m
    a1s = X(t,:); % 1x401
    z2s = a1s * Theta1'; % 1x25 
    a2s = sigmoid(z2s);
    a2s = [ones(1,1) a2s]; % 1x26
    z3s = a2s * Theta2'; % 1x10
    a3s = sigmoid(z3s);
    d3s = a3s - y2(t,:); % 1x10
    d2s = d3s*Theta2.*[ones(1,1) sigmoidGradient(z2s)]; % 10X26, 1X10 1X26
    d2s = d2s(2:end);
    delta2T = delta2T + a2s' * d3s; %25X401
    delta1T = delta1T + a1s' * d2s; %10X26
end

Theta1_gradUR = (1/m) * delta1T';
T1G_reg = (lambda/m)*Theta1;
T1G_reg(:,1) = 0;
Theta1_grad = Theta1_gradUR + T1G_reg;

Theta2_gradUR = (1/m) * delta2T';
T2G_reg = (lambda/m)*Theta2;
T2G_reg(:,1) = 0;
Theta2_grad = Theta2_gradUR + T2G_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
