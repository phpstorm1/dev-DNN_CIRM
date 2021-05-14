function [trained_net, global_optimizer] = train_net(training_feat, training_label, validation_feat, validation_label, net, optimizer, epoch, train_step)

num_net_layer = length(net.layers);
num_samples = size(training_feat,1);
batch_id = genBatchID(num_samples,optimizer.sgd_batch_size);
num_batch = size(batch_id,2);

trained_net = net;
global_optimizer = optimizer;

% initialize optimizer for first traning
if isempty(optimizer.net_weights_inc)
    net_weights_inc = zeroInitNet(net.structure, optimizer.isGPU);
else
    net_weights_inc = optimizer.net_weights_inc;
end
if isempty(optimizer.net_grad_ssqr)
    net_grad_ssqr = zeroInitNet(net.structure, optimizer.isGPU, eps);
else
    net_grad_ssqr = optimizer.net_grad_ssqr;
end
if isempty(optimizer.net_ada_eta)
    net_ada_eta = zeroInitNet(net.structure, optimizer.isGPU);
else
    net_ada_eta = optimizer.net_ada_eta;
end

net_iterative = net.layers;

seq = randperm(num_samples); % randperm dataset every epoch
cost_sum = 0;
for bid = 1:num_batch-1
    perm_idx = seq(batch_id(1,bid):batch_id(2,bid));
    
    if optimizer.isGPU
        % the following two lines are only for mse cost function
        batch_data = gpuArray(training_feat(perm_idx,:));
        batch_label = gpuArray(training_label(perm_idx,:));
    else            
        batch_data = training_feat(perm_idx,:);
        batch_label = training_label(perm_idx,:);
    end
    
    if epoch>optimizer.change_momentum_point
        momentum=optimizer.final_momentum;
    else
        momentum=optimizer.initial_momentum;
    end
    
    %backprop: core code
    [cost,net_grad] = computeNetGradientNoRolling(net_iterative, batch_data, batch_label, optimizer, net.structure);

    %supports only sgd
    for ll = 1:num_net_layer
        switch optimizer.learner
            case 'sgd'
                net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + optimizer.sgd_learn_rate(epoch)*net_grad(ll).W;
                net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + optimizer.sgd_learn_rate(epoch)*net_grad(ll).b;
            case 'ada_sgd'
                net_grad_ssqr(ll).W = net_grad_ssqr(ll).W + (net_grad(ll).W).^2;
                net_grad_ssqr(ll).b = net_grad_ssqr(ll).b + (net_grad(ll).b).^2;
                
                net_ada_eta(ll).W = optimizer.initial_learn_rate./sqrt(net_grad_ssqr(ll).W);                    
                net_ada_eta(ll).b = optimizer.initial_learn_rate./sqrt(net_grad_ssqr(ll).b);
                
                net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + net_ada_eta(ll).W.*net_grad(ll).W;
                net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + net_ada_eta(ll).b.*net_grad(ll).b;
        end
        
        net_iterative(ll).W = net_iterative(ll).W - net_weights_inc(ll).W;
        net_iterative(ll).b = net_iterative(ll).b - net_weights_inc(ll).b;
    end
    cost_sum = cost_sum + cost;
end

validation_predict = predict_from_net(gather(net_iterative), ...
                            gather(validation_feat), ...
                            optimizer);
validation_mse = mse(validation_predict, validation_label);

fprintf('%-20d %-20d %-20.5f %.5f\n', epoch, train_step, cost_sum, validation_mse);

global_optimizer.net_ada_eta = net_ada_eta;
global_optimizer.net_grad_ssqr = net_grad_ssqr;
global_optimizer.net_weights_inc = net_weights_inc;

trained_net.layers = net_iterative;
