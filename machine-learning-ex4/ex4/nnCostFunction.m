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
    X = [ones(m, 1) X];
    
    layer_1_2_activation_value = sigmoid(X * Theta1');
    X_2 = [ones(m, 1) layer_1_2_activation_value];
    h_x = sigmoid(X_2 * Theta2');
    % Before we calculate the cost of the neural network, we need to
    % onehotencode the output label (y) because it is not a binary
    % classification but a 10 label multi-class classification.
    
    one_hot_y = repmat(1:num_labels, m, 1) == repmat(y, 1, num_labels);
    cost = - one_hot_y .* log(h_x) -  (1 - one_hot_y) .* log(1 - h_x);
    J = (1 / m) * (sum(sum(cost)) + (lambda / 2) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))));
    
     triDelta1 = 0;
     triDelta2 = 0;
    
     % For the delta 3
     delta_3 = h_x - one_hot_y;
     % For delta 2
     % Z_2 is the non-sigmoid version of the activation value
     z_2 = [ones(m, 1) X * Theta1'];
     g_z_2 = sigmoidGradient(z_2);
     delta_2 = delta_3 * Theta2 .* g_z_2;
     % This is because we do not find the Gradient for the bias term, se we
     % remove the bias term from the delta
     delta_2=delta_2(:,2:end); % 25 x 5000 
     
     triDelta1 = triDelta1 + delta_2' * X;
     triDelta2 = triDelta2 + delta_3' * X_2;
     
     Theta1_grad = triDelta1 / m;
     Theta2_grad = triDelta2 / m;
     
     Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
     Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);
     % Unroll gradients
     grad = [Theta1_grad(:) ; Theta2_grad(:)];


    end
