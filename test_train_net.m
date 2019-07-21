addpath(genpath('.\\dnn'));

config;
feat_len = 246;

net = gen_net(feat_len, win_len, neighbor_frame, hidden_layer_struct, isGPU);
optimizer = gen_training_optimizer(...
								initial_learn_rate, ...
								every_train_step, ...
								mini_batch_size, ...
								isGPU);

train_data_path = 'D:\coding\Matlab\harm_demo\DATA\factory\tmpdir\db0\feat\train_factory_AmsRastaplpMfccGf.mat';
test_data_path = 'D:\coding\Matlab\harm_demo\DATA\factory\tmpdir\db0\feat\test_factory_AmsRastaplpMfccGf.mat';

load(train_data_path);
train_feat = feat_data;
train_label = feat_label;
validation_protion = 0.1;
validation_size = ceil(size(train_label, 1) * validation_protion);
validation_feat = train_feat(1:validation_size, :);
validation_label = train_label(1:validation_size, :);
train_feat(1:validation_size, :) = [];
train_label(1:validation_size, :) = [];

[norm_train_feat, mu_batch_feat, std_batch_feat] = mean_var_norm(train_feat);
train_feat_win = win_buffer(norm_train_feat, neighbor_frame);

validation_feat_norm = mean_var_norm_testing(...
								validation_feat, ...
								mu_batch_feat, ...
								std_batch_feat);
validation_feat_win = win_buffer(validation_feat_norm, neighbor_frame);

load(test_data_path);
test_feat = feat_data;
test_label = feat_label;

test_feat_norm = mean_var_norm_testing(...
								test_feat, ...
								mu_batch_feat, ...
								std_batch_feat);
test_feat_win = win_buffer(test_feat_norm, neighbor_frame);

[net, optimizer] = train_net(train_feat_win, train_label, test_feat_win, test_label, net, optimizer);