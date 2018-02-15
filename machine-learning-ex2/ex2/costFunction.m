function [J, grad] = costFunction(theta, X, y)
    %COSTFUNCTION Compute cost and gradient for logistic regression
    %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    %   parameter for logistic regression and the gradient of the cost
    %   w.r.t. to the parameters.

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly 
    % J = 0;
    % grad = zeros(size(theta));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    %
    % Note: grad should have the same dimensions as theta
    
    % Logistic Regression h(x)..Sigmoid conversion of Linear Regression
    h_x = sigmoid(X * theta);
    cost_for_all_points_part_1 = transpose(y) * log(h_x);  % This gives a scalar
    cost_for_all_points_part_2 = transpose(1 - y) * log(1 - h_x);  % This also gives a scalar
    cost_for_all_points = cost_for_all_points_part_1 + cost_for_all_points_part_2;
    % Now return the actual cost
    
    J = -(cost_for_all_points / m);  % This is also a scalar
    
    % It is the same as the linear regression with different h_x
    % The transpose and multiplication takes care for all the 'm' test
    % points and hence not summition is required. True meaning of
    % vectorized operation.
    grad = 1 / m * transpose(X) * (h_x - y);
    % =============================================================
end
