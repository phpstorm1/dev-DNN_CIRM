function [ training_label ] = get_cirm_labels(	 voice_sig, ...
												 mix_sig, ...
												 win_len, ...
												 win_shift, ...
												 fs, ...
												 q, ...
												 c )

%Returns a harmonic training label matrix for a voiced speech
%	Input:
%	voice_sig: single-channel clean speech
%	mix_sig: single-channel noisy speech
%	win_len: window length for framing
%	win_shift: shifts between window
%	fs: sampling frequency
%	use_fixed_sca: flag inficating whether fixed scaling factor is used
%	sca_fac: fiexed scaling factor for residual
%	Output:
%	training_label: the magnitude of STFT

%win_fun = sqrt(hanning(win_len, 'periodic'));
%win_fun = hamming(win_len, 'periodic');
win_fun = sqrt(hann(win_len, 'periodic'));
%win_fun = rectwin(win_len);

voice_sig = voice_sig(:)';
voice_sig = voice_sig / max(abs(voice_sig));
mix_sig = mix_sig / max(abs(mix_sig));

spch_frame = enframe(voice_sig, win_fun, win_shift);
%spch_frame = frame_sig(voice_sig, win_len, win_shift, @(x)ones(x,1));
freq_frame = rfft(spch_frame')';

mix_frame = enframe(mix_sig, win_fun, win_shift);
%mix_frame = frame_sig(mix_sig, win_len, win_shift, @(x)ones(x,1));
mix_freq = rfft(mix_frame')';

cirm = zeros(size(mix_freq,1), size(mix_freq,2)*2);
% for i=1:size(cirm,1)
% 	for j=1:size(cirm,2)
% 		if j<=size(mix_freq,2)
% 			cirm(i,j) = (real(mix_freq(i,j))*real(freq_frame(i,j)) + ...
% 				imag(mix_freq(i,j))*imag(freq_frame(i,j))) / abs(mix_freq(i,j));
%         else
%             col_idx = j - size(mix_freq,2);
% 			cirm(i,j) = (real(mix_freq(i,col_idx))*imag(freq_frame(i,col_idx)) - ...
% 				real(mix_freq(i,col_idx))*imag(freq_frame(i,col_idx))) / abs(mix_freq(i,col_idx));
% 		end
% 		cirm(i,j) = q*((1-exp(-c*cirm(i,j)))/(1+exp(-c*cirm(i,j))));
% 	end
% end
cirm(:, 1:size(mix_freq,2)) = real(freq_frame./mix_freq);
cirm(:, size(mix_freq,2)+1:size(cirm,2)) = imag(freq_frame./mix_freq);

% for m=1:size(cirm,1)
% 	for n=1:size(cirm,2)
% 		cirm(m,n) = q*(1-exp(-c*cirm(m,n)))/(1+exp(-c*cirm(m,n)));
% 	end
% end

denom = 1+exp(-c*cirm);
nom = 1-exp(-c*cirm);
cirm = q*nom./denom;


training_label = cirm;
training_label = training_label';

end

