function plotData(X, y)
    %PLOTDATA Plots the data points X and y into a new figure 
    %   PLOTDATA(x,y) plots the data points with + for the positive examples
    %   and o for the negative examples. X is assumed to be a Mx2 matrix.

    % Create New Figure
    figure('position', [100, 100, 900, 700]); hold on;

    % ====================== YOUR CODE HERE ======================
    % Instructions: Plot the positive and negative examples on a
    %               2D plot, using the option 'k+' for the positive
    %               examples and 'ko' for the negative examples.
    %
    
    % select the 0 label index
%    label_0 = find(y == 0);
    % select the 1 label index
 %   label_1 = find(y == 1);
    
    % Plotting for labels 0
    plot(X(y == 0, 1), X(y == 0, 2), 'ko', 'LineWidth', 1, 'MarkerFaceColor', 'yellow', 'MarkerSize', 8);
    % Plotting for the labels 1
    plot(X(y == 1, 1), X(y == 1, 2), 'k+', 'LineWidth', 3, 'MarkerSize', 10);
    % =========================================================================
    hold off;
end
