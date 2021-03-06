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
y = y == [1:10];
X = [ones(m, 1) X];
for i = 1:m
    for k = 1:num_labels
        z2 = Theta1 * X(i, :)';
        a2 = sigmoid(z2);
        a2 = [ones(1, size(a2, 2)); a2];
        z3 = Theta2 * a2;
        a3 = sigmoid(z3);
        h = a3(k, :);
        yt = y(i, k);
        J = J + ((-1 * yt) * log(h) - (1 - yt) * log(1 - h));
    end;
end;
J = (1 / m) * J;


reg = 0;


Theta1t = Theta1(:, 2:end);
t1 = 0;
for j = 1:hidden_layer_size
    for k = 1:input_layer_size
        t1 = t1 + Theta1t(j, k) ^ 2;
    end;
end;


Theta2t = Theta2(:, 2:end);
t2 = 0;
for j = 1:num_labels
    for k = 1:hidden_layer_size
        t2 = t2 + Theta2t(j, k) ^ 2;
    end;
end;

reg = (lambda / (2 * m)) * (t1 + t2);

J = J + reg;


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

D1 = 0;
D2 = 0;
d3 = zeros(num_labels, 1);
for t=1:m
    a1 = X(t, :);
    
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);
    a2 = [ones(1, size(a2, 2)); a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    for k=1:num_labels
        yk = y(t, :)(:, k);
        d3(k, 1) = a3(k, 1) - yk;

    end;
    d2 = (Theta2(:,2:end)' * d3) .* sigmoidGradient(z2);
    D2 = D2 + (d3 * a2');
    D1 = D1 + (d2 * a1);
end;


Theta1_grad = D1 / m;
Theta2_grad = D2 / m;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
