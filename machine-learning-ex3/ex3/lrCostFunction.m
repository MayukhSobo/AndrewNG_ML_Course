function [J, grad] = lrCostFunction(theta, X, y, lambda)
  %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  %regularization
  %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
  %

  % Cost computation
  h_x = sigmoid(X * theta);
  non_reg_cost = (-1.0 / m) * ( (transpose(y) * log(h_x)) + (transpose(1 - y) * log(1 - h_x)) );
  reg_term = (lambda / (2 * m)) * sum(theta(2:end, 1).^2);
  J = non_reg_cost + reg_term;
  
  % Gradient Computation
  non_reg_gradient = (1.0 / m) * transpose(X) * (h_x - y);
  theta_0 =  non_reg_gradient(1, 1);
  grad = non_reg_gradient + (lambda / m) * theta;
  grad(1, 1) = theta_0;
  % grad = grad(:);
end
