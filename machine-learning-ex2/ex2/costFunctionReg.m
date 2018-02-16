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
  
  % The cost is the sum of normal cost and regularised value. We have already
  % calculated the normal cost. Only regularized term needs to be added.
  
  [normal_cost, normal_gradient] = costFunction(theta, X, y);
  
  % For all the theta except the theta_0
  regularized_cost = (lambda / (2 * m)) * sum(theta(2:end, 1).^2);
  
  % Now calculate the total cost
  J = normal_cost + regularized_cost;
  % Calculate gradient for all theta including theta_0
  theta_0 = normal_gradient(1, 1);
  grad = normal_gradient + (lambda / m) * theta;

  % Now remove the regularization from theta_0
  grad(1, 1) = theta_0;
  % =============================================================
end
