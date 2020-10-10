function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



h_theta = X * theta;
J = sum((h_theta - y) .^ 2) / 2 / m;
J = J + lambda / 2 / m * sum(theta(2:end, :) .^ 2);

rep_h_theta = repmat((h_theta - y), 1, size(X, 2));
mul_h_x = rep_h_theta .* X;
grad = sum(mul_h_x, 1) / m;
grad = grad';
grad(2:end) = grad(2:end) + lambda * theta(2:end) / m;





% =========================================================================

grad = grad(:);

end
