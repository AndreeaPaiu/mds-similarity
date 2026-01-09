% 1. Load the JSON file
fileName = 'mds_100_tests.json';
jsonStr = fileread(fileName);
data = jsondecode(jsonStr);

% 2. MDS parameters (matching your previous setup)
MDSparam.iter       = 400;
MDSparam.verbose    = 'off'; % 'off' for speed during bulk tests
MDSparam.xhistory   = 'off';
MDSparam.rtol       = 1e-6;
MDSparam.atol       = 1e-3;

% 3. Loop through each test case
for i = 1:length(data)
    test = data(i);
    
    % Extract inputs
    X_targets_real = test.inputs.X_targets_real;
    X_anchors = test.inputs.X_anchors;
    X0 = test.inputs.X0;
    n_anchors = test.config.n_anchors;
    n_targets = test.config.n_targets;
    w_tt = test.inputs.w_tt;
    w_ta = test.inputs.w_ta;
    nodes = n_targets + n_anchors;
    
    % Generate Ground Truth Distance Matrix (D)
    X_real = [X_targets_real; X_anchors];
    D = pdist2(X_real, X_real);
    
    % Generate Weight Matrix (W)
    W = ones(nodes, nodes);
    W(1:n_targets, 1:n_targets) = w_tt;
    W(1:n_targets, n_targets+1:end) = w_ta;
    W(n_targets+1:end, 1:n_targets) = w_ta;
    W(n_targets+1:end, n_targets+1:end) = 0; % Anchors fixed
    W(logical(eye(nodes))) = 0; % Set diagonal to 0
    
    % 4. Run your smacofAnchors function
    [X_est, hist] = smacofAnchors(D, X0, W, n_anchors, MDSparam);
    
    % 5. Save results back into the structure
    data(i).results.matlab_X_est = X_est;
    data(i).results.matlab_stress = hist.s(end);
    
    fprintf('Processed test %d/%d\n', i, length(data));
end

% 6. Save the updated data back to JSON
newJsonStr = jsonencode(data, 'PrettyPrint', true);
fid = fopen(fileName, 'w');
fprintf(fid, '%s', newJsonStr);
fclose(fid);

disp('Successfully filled MATLAB results for 100 tests.');