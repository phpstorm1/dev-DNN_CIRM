function print_eval_res(fid, eval, wav_type, noise_type, snr, list_testing, print_avg)
%print_eval_res Print the evaluation result on screen and wirte it into the file
% Input
%	fid: fileID for writing the evaluation result
%	eval: a structure contains evaluation result. Composed of various martix
%	wav_type: an indicator of wether the wav is noisy, label, or predicted labels
%	noise_type: a vector containing noise name in string
%	snr: a vector containing real-value snr
% 	list_testing: a vector containing filenames for the testing
%	print_avg: a binary flag indicating wether to print the average result or result
%				on all testing files
if strcmp(lower(wav_type), 'mix')
	fprintf(fid, '\nMix\n');
	fprintf('\nMix\n');
	wav_type = 1;
elseif strcmp(lower(wav_type), 'ideal')
	fprintf(fid, '\nIdeal\n');
	fprintf('\nIdeal\n');
	wav_type = 2;
elseif strcmp(lower(wav_type), 'estimated')
	fprintf(fid, '\nEstimated\n');
	fprintf('\nEstimated\n');
	wav_type = 3;
else
	error("Unrecognized wav_type. \n")
end

mean_pesq = zeros(numel(snr), numel(noise_type));
mean_stoi = zeros(numel(snr), numel(noise_type));
mean_ssnr = zeros(numel(snr), numel(noise_type));

% write to file
if ~print_avg
	fprintf(fid, '%-20s %-20s %-20s %-20s %-20s %s\n', 'wav_ID', 'nosise', 'snr', ...
												'pesq', 'stoi', 'ssnr');
	jump_idx = numel(list_testing) * numel(noise_type);
	for snr_idx=1:numel(snr)
		for noise_idx=1:numel(noise_type)
			cur_snr = snr(snr_idx);
			cur_noise = noise_type(noise_idx);
			for testing_item=1:numel(list_testing)
				print_idx = testing_item + (noise_idx-1)*numel(list_testing) + ...
								(snr_idx - 1) * jump_idx;
				fprintf(fid, '%-20s %-20s %-20s %-20s %-20s %s\n', ...
								num2str(testing_item), ...
								cur_noise, ...
								num2str(cur_snr), ...
								num2str(eval.pesq(print_idx, wav_type)), ...
								num2str(eval.stoi(print_idx, wav_type)), ...
								num2str(eval.ssnr(print_idx, wav_type)));
			end
		end
	end
else
	fprintf(fid, '%-20s %-20s %-20s %-20s %s\n', 'nosise', 'snr', ...
											'pesq', 'stoi', 'ssnr');
	jump_idx = numel(noise_type);
	for snr_idx=1:numel(snr)
		for noise_idx=1:numel(noise_type)
			cur_snr = snr(snr_idx);
			cur_noise = noise_type(noise_idx);
			print_idx = noise_idx + (snr_idx - 1) * jump_idx;
			fprintf(fid, '%-20s %-20s %-20s %-20s %s\n', ...
							cur_noise, ...
							num2str(cur_snr), ...
							num2str(eval.avg_pesq(print_idx, wav_type)), ...
							num2str(eval.avg_stoi(print_idx, wav_type)), ...
							num2str(eval.avg_ssnr(print_idx, wav_type)));
			mean_pesq(snr_idx, noise_idx) = eval.avg_pesq(print_idx, wav_type);
			mean_ssnr(snr_idx, noise_idx) = eval.avg_ssnr(print_idx, wav_type);
			mean_stoi(snr_idx, noise_idx) = eval.avg_stoi(print_idx, wav_type);

		end
	end

	fprintf(fid, '------------------------------------------------------\n');
	fprintf(fid, '%-20s %-20s %-20s %s\n', 'snr', 'pesq', 'stoi', 'ssnr');
	for snr_idx=1:numel(snr)
		cur_snr = snr(snr_idx);
		fprintf(fid, '%-20s %-20s %-20s %s\n', ...
						num2str(cur_snr), ...
						num2str(mean(mean_pesq(snr_idx, :))), ...
						num2str(mean(mean_stoi(snr_idx, :))), ...
						num2str(mean(mean_ssnr(snr_idx, :))));
	end

	fprintf(fid, '------------------------------------------------------\n');
	fprintf(fid, '%-20s %-20s %-20s %s\n', 'noise', 'pesq', 'stoi', 'ssnr');
	for noise_idx=1:numel(noise_type)
		cur_noise = noise_type(noise_idx);
		fprintf(fid, '%-20s %-20s %-20s %s\n', ...
						cur_noise, ...
						num2str(mean(mean_pesq(:, noise_idx))), ...
						num2str(mean(mean_stoi(:, noise_idx))), ...
						num2str(mean(mean_ssnr(:, noise_idx))));
	end

	fprintf(fid, '------------------------------------------------------\n');
	fprintf(fid, '%-20s %-20s %-20s %s\n', 'avg', 'pesq', 'stoi', 'ssnr');
	fprintf(fid, '%-20s %-20s %-20s %s\n', ...
				' ', ...
				num2str(mean(mean(mean_pesq))), ...
				num2str(mean(mean(mean_stoi(:, noise_idx)))), ...
				num2str(mean(mean(mean_ssnr(:, noise_idx)))));
end

% print on screen
if ~print_avg
	fprintf('%-20s %-20s %-20s %-20s %-20s %s\n', 'wav_ID', 'nosise', 'snr', ...
												'pesq', 'stoi', 'ssnr')
	jump_idx = numel(list_testing) * numel(noise_type);
	for snr_idx=1:numel(snr)
		for noise_idx=1:numel(noise_type)
			cur_snr = snr(snr_idx);
			cur_noise = noise_type(noise_idx);
			for testing_item=1:numel(list_testing)
				print_idx = testing_item + (noise_idx-1)*numel(list_testing) + ...
								(snr_idx - 1) * jump_idx;
				fprintf('%-20s %-20s %-20s %-20s %-20s %s\n', ...
						num2str(testing_item), ...
						cur_noise, ...
						num2str(cur_snr), ...
						num2str(eval.pesq(print_idx, wav_type)), ...
						num2str(eval.stoi(print_idx, wav_type)), ...
						num2str(eval.ssnr(print_idx, wav_type)));
			end
		end
	end
else
	fprintf('%-20s %-20s %-20s %-20s %s\n', 'nosise', 'snr', ...
											'pesq', 'stoi', 'ssnr')
	jump_idx = numel(noise_type);
	for snr_idx=1:numel(snr)
		for noise_idx=1:numel(noise_type)
			cur_snr = snr(snr_idx);
			cur_noise = noise_type(noise_idx);
			print_idx = noise_idx + (snr_idx - 1) * jump_idx;
			fprintf('%-20s %-20s %-20s %-20s %s\n', ...
					cur_noise, ...
					num2str(cur_snr), ...
					num2str(eval.avg_pesq(print_idx, wav_type)), ...
					num2str(eval.avg_stoi(print_idx, wav_type)), ...
					num2str(eval.avg_ssnr(print_idx, wav_type)));
		end
	end
	
	fprintf('------------------------------------------------------\n');
	fprintf('%-20s %-20s %-20s %s\n', 'snr', 'pesq', 'stoi', 'ssnr');
	for snr_idx=1:numel(snr)
		cur_snr = snr(snr_idx);
		fprintf('%-20s %-20s %-20s %s\n', ...
						num2str(cur_snr), ...
						num2str(mean(mean_pesq(snr_idx, :))), ...
						num2str(mean(mean_stoi(snr_idx, :))), ...
						num2str(mean(mean_ssnr(snr_idx, :))));
	end

	fprintf('------------------------------------------------------\n');
	fprintf('%-20s %-20s %-20s %s\n', 'noise', 'pesq', 'stoi', 'ssnr');
	for noise_idx=1:numel(noise_type)
		cur_noise = noise_type(noise_idx);
		fprintf('%-20s %-20s %-20s %s\n', ...
						cur_noise, ...
						num2str(mean(mean_pesq(:, noise_idx))), ...
						num2str(mean(mean_stoi(:, noise_idx))), ...
						num2str(mean(mean_ssnr(:, noise_idx))));
	end

	fprintf('------------------------------------------------------\n');
	fprintf('%-20s %-20s %-20s %s\n', 'avg', 'pesq', 'stoi', 'ssnr');
	fprintf('%-20s %-20s %-20s %s\n', ...
				' ', ...
				num2str(mean(mean(mean_pesq))), ...
				num2str(mean(mean(mean_stoi(:, noise_idx)))), ...
				num2str(mean(mean(mean_ssnr(:, noise_idx)))));
end

end

