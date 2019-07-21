function [ syn_wav ] = wav_synthesis( label, mix, fs, win_len, win_shift, q, c )
%Returns a synthesised speech using the output from NN
% Input:
%	label: output from DNN, matrix containing amplititude and IF (phase)
%	fs: sampling frequency
%	win_len: window length
%	win_shift: shift between windows

win_fun = sqrt(hann(win_len, 'periodic'));
%win_fun = rectwin(win_len);

% use mix phase for generating the waveform
mix_frame = enframe(mix, win_fun, win_shift);
%mix_frame = frame_sig(mix, win_len, win_shift, @(x)ones(x,1));

mix_fft = rfft(mix_frame')';

label = label';
num_col_split = size(label,2)/2;
num_frame = size(label, 1);

% rec_label = zeros(size(label));
% for i=1:size(label,1)
% 	for j=1:size(label,2)
% 		rec_label(i, j) = - (1/c) * log((q-label(i,j))/(q+label(i,j)));
% 	end
% end
log_nom = q - label;
log_denom = q + label;
log_label = log(log_nom ./ log_denom);
rec_label = -(1/c) * log_label;

mask = rec_label(:, 1:num_col_split) + (1i)*rec_label(:, num_col_split+1:size(label,2));

dft_frame = mask .* mix_fft;

wav_frame = real(irfft(dft_frame')');

thres = 1;
wav_frame(abs(wav_frame)>=thres) = 0;

syn_wav = overlapadd(wav_frame, win_fun, win_shift);
%syn_wav = deframe_sig(wav_frame, [], win_len, win_shift, @(x)ones(x,1));
syn_wav = syn_wav./max(abs(syn_wav));

end
