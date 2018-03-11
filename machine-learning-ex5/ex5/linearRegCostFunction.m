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
    h_x = X * theta;
    residual = (h_x - y) .^ 2;
    non_reg_cost = sum(residual);
    reg_cost = theta(2:end, :) .^ 2;
    total_reg_cost = sum(reg_cost);
    
    J = ((1 / (2 * m)) * non_reg_cost) + ((lambda / (2 * m)) * total_reg_cost);
    % =========================================================================
    
    grad = (1 / m) * sum((h_x - y) .* X);
    reg_term = (lambda / m) * theta(2:end);
    grad(:, 2:end) = grad(:, 2:end) + reg_term';
%     size(grad)
%     grad = sum(grad);
    
    grad = grad(:);

    end
