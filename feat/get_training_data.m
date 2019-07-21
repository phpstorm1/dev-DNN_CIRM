function [ feature, label ] = get_training_data( clean_wav,...
												 noisy_wav,...
												 feat_fun,...
												 win_len,...
												 win_shift,...
												 fs,...
												 q,...
												 c)
%get_training_data Obtain the training feature and training labels
%   Input:
%	clean_wav: single-channel clean speech
%	noisy_wav: single-channel noisy speech
%	feat_fun: function name for obtaining the feature
%	win_len: length of analytical window
%	win_shift: shift between adjacent frames
%	fs: sampling frequency
%	useFixedScaFac: binary flag indicating whether to use fixed scaling factor 
%					when obtaining labels
%	sca_fac: the constant for obtaining training labels if useFixedScaFac is nonzero
%	Output:
%	feature: training feature obtained from noisy speech
%	label: training label obtained from clean speech

label = get_cirm_labels(clean_wav, noisy_wav, win_len, win_shift, fs, ...
						 q, c);

feature = feval(feat_fun, noisy_wav, win_len, win_shift);

if(size(feature, 2) ~= size(label, 2))
	feature = feature(:, 1:min(size(feature, 2), size(label, 2)));
	label = label(:, 1:min(size(feature,2), size(label, 2)));
end

end