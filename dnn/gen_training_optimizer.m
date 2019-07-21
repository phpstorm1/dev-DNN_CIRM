function [ optimizer ] = gen_training_options( learn_rate, max_epoch, batch_size , isGPU )
%gen_training_options define the optimizer for training
% Input:
%	learn_rate:
%	max_epoch:
%	batch_size:
% Output:
%	options:	

optimizer = [];
optimizer.learner = 'ada_sgd';

optimizer.initial_learn_rate = learn_rate;
optimizer.sgd_max_epoch = max_epoch;
optimizer.sgd_batch_size = batch_size;
optimizer.isGPU = isGPU;
optimizer.eval_on_gpu = 0;

optimizer.initial_momentum = 0.5;
optimizer.final_momentum = 0.9;
optimizer.change_momentum_point = 5;

optimizer.unit_type_hidden = 'relu';
optimizer.unit_type_output = 'lin';
optimizer.cost_function = 'mse';
optimizer.isDropout = 0;
optimizer.isDropoutInput = 0;
optimizer.drop_ratio = 0.15;

optimizer.net_weights_inc = [];
optimizer.net_grad_ssqr = [];
optimizer.net_ada_eta = [];

end
