clc;
clear;
close all;

% MDS parameters
MDSparam.iter       = 400;
MDSparam.verbose    = 1;
MDSparam.xhistory   = 'off';
MDSparam.rtol       = 1e-6;
MDSparam.atol       = 1e-3;

% Anchors (known)
X_anchors = [
    6 6 0;
    4 4 0;
    5 5 0;
    1 3 0;
    2 4 0;
];

% Targets (unknown, simulate)
X_targets = [
    0 0 0;
    3 3 0;
    2 2 0;
    1 1 0;
];

% Full real positions: 4 targets + 3 anchors
X_real = [X_targets; X_anchors];
disp('Matrice puncte reale')
disp(X_real)

% Parameters
n_targets = size(X_targets, 1);
n_anchors = size(X_anchors, 1);
nodes = n_targets + n_anchors;
dimension = 3;

% Distance matrix between all nodes
D = pdist2(X_real, X_real);
disp('Matrice distante')
disp(D)

W = ones(nodes, nodes);
W(logical(eye(nodes))) = 0;
W(1:n_targets, 1:n_targets) = 0.2; 
W(1:n_targets, n_targets+1:end) = 0.7;
W(n_targets+1:end, 1:n_targets) = 0.7;
W(logical(eye(nodes))) = 0;
disp('matrice greutati')
disp(W)



% Initial guess: targets random, anchors fixed
X0 = [rand(n_targets, dimension)*10; X_anchors];

% Run MDS with anchors
[X_est, hist] = smacofAnchors(D, X0, W, n_anchors, MDSparam);
disp(X_est)

% Plot results
figure;
hold on;
plot3(X_targets(:,1), X_targets(:,2), X_targets(:,3), 'bo', 'MarkerSize', 10, 'DisplayName', 'True Targets');
plot3(X_est(:,1), X_est(:,2), X_est(:,3), 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Estimated Targets');
plot3(X_anchors(:,1), X_anchors(:,2), X_anchors(:,3), 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Anchors');
legend;