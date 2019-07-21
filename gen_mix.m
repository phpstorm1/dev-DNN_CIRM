function [ mix_wav ] = gen_mix( clean_wav, noise_wav, snr )
%gen_mix Mix speech and noise by given SNR. Only single-channel wav is supported
%   Input:
%	clean_wav: a single-channel clean speech
%	noise_wav: a single-channel noise
%	snr: signal-to-noise ratio. 
%	Output:
%	mix_wav: a single-channel noisy speech

% reshape to row vectors
clean_wav = clean_wav(:)';
noise_wav = noise_wav(:)';

%{
if is_training
	noise_wav = noise_wav(1:floor(length(noise_wav)/2));
else
	noise_wav = noise_wav(floor(length(noise_wav)/2)+1:end);
end
%}

% normalize to [-1, 1]
%clean_wav = clean_wav / max(abs(clean_wav));
%noise_wav = noise_wav / max(abs(noise_wav));

len_cl = length(clean_wav);
len_noise = length(noise_wav);

% if noise is longer than speech, repeat the noise
rep_noise = noise_wav;
if len_cl > len_noise
	rep_noise = repmat(noise_wav, [1, ceil(len_cl/len_noise)]);
end
rep_noise = [rep_noise noise_wav];

% randomly select a part of the noise
begin_cut = randi(len_noise);
tmp_noise = rep_noise(begin_cut:begin_cut+len_cl-1);

cur_snr = 10*log10(sum(clean_wav.^2)/sum(tmp_noise.^2));
alpha1 = sqrt(sum(clean_wav.^2)/(sum(tmp_noise.^2)*10^(snr/10)));

mix_wav = clean_wav + alpha1*tmp_noise;
mix_wav = mix_wav / max(abs(mix_wav));


end