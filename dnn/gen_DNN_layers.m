function [ layers ] = gen_DNN_layers( feat_len, win_len, neighbor_frame )
%gen_DNN Generate DNN
% Input:
%	feat_len: length of a single-frame feature
%	win_len: length of analytical window
%	neighbor_frame: how many adjacent frames to use
% Output:
%	layers: structured fully-connected layer

output_size = round(win_len/2) + 1;
input_size = feat_len * (neighbor_frame*2 + 1);

layers = [ ...
	sequenceInputLayer(input_size)
	fullyConnectedLayer(512)
	reluLayer
	fullyConnectedLayer(512)
	reluLayer
	fullyConnectedLayer(512)
	reluLayer
	fullyConnectedLayer(output_size)
	regressionLayer
	];

end


