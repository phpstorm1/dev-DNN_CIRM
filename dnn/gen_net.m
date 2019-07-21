function [ net ] = gen_net( feat_len, ... 
							win_len, ...
							adjacent_frame , ...
							hidden_layer_struct, ...
							useInputNormalization, ...
							isGPU )
%gen_DNN Generate DNN
% Input:
%	feat_len: length of a single-frame feature
%	win_len: length of analytical window
%	adjacent_frame: how many adjacent frames to use
%	hidden_layer_struct: tavector indicating the structure of the hidden layers
%	useInputNormalization: a binary flag indicating wether to use input normalization
%	isGPu: a binary flag indicating wether to use GPU for training
% Output:
%	layers: structured fully-connected layer
output_size = round(win_len/2+1)*2;
input_size = feat_len * (adjacent_frame*2 + 1);

net = [];
% define the structure of DNN
net.structure = [input_size, hidden_layer_struct, output_size];
net.adjacent_frame = adjacent_frame;
net.useInputNormalization = useInputNormalization;
net.norm_mu = [];
net.norm_std = [];

% initialize the layers
isSparse = 0;
isNorm = 1;
net.layers = randInitNet(net.structure, isSparse, isNorm, isGPU);

end


