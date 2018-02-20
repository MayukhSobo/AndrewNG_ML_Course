function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values

% num_labels = size(Theta2, 1);

% For layer 1 to 2 
m = size(X, 1);
X = [ones(m, 1) X];
layer_1_2_weights = sigmoid(Theta1 * X');

% For layer 2 to 3 (output layer)
m = size(layer_1_2_weights, 2);
X = [ones(1, m); layer_1_2_weights];
layer_2_3_weights = sigmoid(Theta2 * X);
%disp(max(layer_2_3_weights))
[~, p] = max(layer_2_3_weights);
p = transpose(p);
%pause
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
