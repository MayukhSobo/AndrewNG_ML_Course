function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
      %GRADIENTDESCENT Performs gradient descent to learn theta
      %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
      %   taking num_iters gradient steps with learning rate alpha
      % Initialize some useful values
      m = length(y); % number of training examples
      J_history = zeros(num_iters, 1);
      for iter = 1:num_iters
      % ====================== YOUR CODE HERE ======================
      % Instructions: Perform a single gradient step on the parameter vector
      %               theta. 
      %
      % Hint: While debugging, it can be useful to print out the values
      %       of the cost function (computeCost) and gradient here.
      %
      h_x = X * theta;
      delta = transpose(h_x - y) * X; % This transpose is to convert 97x1 matrix 
                                      % into 1x97 for further multiplication
      theta = theta - (alpha / m) .* delta'; % This transpose is to convert 1x2 matrix into 2x1 matrix
      % Finally we need a 2x1 matrix
      % This solution should be generic
      % ============================================================
      % Save the cost J in every iteration    
      J_history(iter) = computeCost(X, y, theta);
  end
end