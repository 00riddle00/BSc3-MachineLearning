#!/usr/bin/env octave

% make a grid
[w1, w2] = meshgrid(- 5:0.1:5, - 5:0.1:5);

% fix the weight of the bias
w0 = 1;

% system of inequalities
ineq1 = - w0 - 0.2 * w1 + 0.5 * w2 <= 0;
ineq2 = - w0 + 0.2 * w1 - 0.5 * w2 <= 0;
ineq3 = - w0 + 0.8 * w1 - 0.8 * w2 > 0;
ineq4 = - w0 + 0.8 * w1 + 0.8 * w2 > 0;

% color palette (black color (=0 0 0) at the last
% position indicates where the solutions reside)
mymap = [1 1 1; 1 1 1; 1 1 1; 0 0 0];
colormap(mymap);
colors = zeros(size(w0)) + ineq1 + ineq2 + ineq3 + ineq4;

% draw the system of inequalities solution graph
pl0t = scatter(w1(:), w2(:), 3, colors(:), 'filled');

waitfor(pl0t)
disp('Exiting...')
