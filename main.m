addpath(genpath('./feat'));
addpath(genpath('./dnn'));

% load configurations
config;

% write log file
t = datestr(now, 'mmddHHMM');
if ~exist(save_path)
	mkdir(save_path)
end
diary([save_path, filesep, 'main.', t, '.log']);
diary on;

fprintf('noise: %s \n', noise_type);
fprintf('snr: %s \n', num2str(snr));
fprintf('win_len: %d \n', win_len);
fprintf('win_shift: %d \n', win_shift);
fprintf('file_per_batch: %d\n', file_per_batch);
fprintf('hidden_layer_struct: %s\n', num2str(hidden_layer_struct));
fprintf('learn_rate: %f \n', initial_learn_rate);
fprintf('adjacent_frame: %d \n', adjacent_frame);
fprintf('mini_batch_size: %d \n', mini_batch_size);
fprintf('-------------------------------------------------------\n')

if ~read_from_list

	speech_file_list = dir([speech_path, filesep, '**', filesep,'*.wav']);
	num_files = numel(speech_file_list);

	% get validation file list
	len_validation = round(num_files*validation_percentage/100);
	if ~len_validation
		error('validation list is empty');
	end
	list_validation = speech_file_list(1:len_validation);

	% get test file list 
	list_testing = dir([test_path, filesep, '**', filesep,'*.wav']);
	len_testing = numel(list_testing);
	if ~len_testing
		error('testing list is empty');
	end
	
	% get training file list
	list_training = speech_file_list(len_validation+1:num_files);
	len_training = numel(list_training);
	if ~len_training
		error('training list is empty');
	end

	% get noise file list
	list_noise = dir([noise_path, filesep, '**', filesep,'*.wav']);
	len_noise = numel(list_noise);
	if ~len_noise
		error('noise list is empty');
	end

else

	fid_train_txt = fopen(train_list_path);
	fid_test_txt = fopen(test_list_path);

	tmp = textscan(fid_train_txt, "%s");
    num_files = length(tmp{1});
    for i=1:num_files
        speech_file_list(i).name = tmp{1}(i);
    end

	len_validation = round(num_files*validation_percentage/100);
	if ~len_validation
		error('validation list is empty');
	end
	list_validation = speech_file_list(1:len_validation);

	list_training = speech_file_list(len_validation+1:num_files);
	len_training = numel(list_training);
	if ~len_training
		error('training list is empty');
	end

	tmp = textscan(fid_test_txt, "%s");
	len_testing = length(tmp{1});
    for i=1:len_testing
        list_testing(i).name = tmp{1}(i);
    end

	if ~len_testing
		error('testing list is empty');
	end

	% get noise file list
	list_noise = dir([noise_path, filesep, '**', filesep,'*.wav']);
	len_noise = numel(list_noise);
	if ~len_noise
		error('noise list is empty');
	end

end

validation_wav = cell(len_validation, 1);
% read validation files
fprintf('reading validation .wav files...');
for validation_idx=1:len_validation
	[validation_wav{validation_idx}, fs0] = audioread(...
			[speech_path, filesep, char(list_validation(validation_idx).name)]);
	% resample to 16k Hz
	validation_wav{validation_idx} = resample(...
				validation_wav{validation_idx}, 16e3, fs0);

	if ~isempty(wav_max_len)
		max_len = wav_max_len * 16e3;
		if length(validation_wav{validation_idx}) > max_len
			validation_wav{validation_idx} = validation_wav{validation_idx}(1:max_len);
		else
% 			validation_wav{validation_idx} = [validation_wav{validation_idx};
%                      zeros(max_len-length(validation_wav{validation_idx}), 1)];
            validation_wav{validation_idx} = padarray(validation_wav{validation_idx}, ...
                            max_len - length(validation_wav{validation_idx}),...
                            'post');
		end
	end

    validation_wav{validation_idx} = validation_wav{validation_idx} / ...
                                max(abs(validation_wav{validation_idx}));
end
fprintf('...done\n');

% read noise files
fprintf('reading noise .wav files...');

noise_wav=cell(len_noise, 1);
validation_noise_wav = cell(numel(noise_type), 1);
validation_noise_idx = 1;
for noise_idx=1:len_noise
	[noise_wav{noise_idx}, fs0] = audioread(...
				[list_noise(noise_idx).folder, filesep, list_noise(noise_idx).name]);
	noise_wav{noise_idx} = resample(noise_wav{noise_idx}, 16e3, fs0);
	for noise_type_idx=1:numel(noise_type)
		if strcmp(list_noise(noise_idx).name, [char(noise_type(noise_type_idx)), '.wav'])
			validation_noise_wav{validation_noise_idx} = resample(noise_wav{noise_idx}, 16e3, fs0);
			validation_noise_idx = validation_noise_idx + 1;
		end
	end
end
test_noise_wav = validation_noise_wav;

fprintf('...done\n')

% add noise to validation, get the feature and labels
fprintf('obtaining validation data...');
if ~load_validation_data
	validation_feat = [];
	validation_label = [];
	for snr_idx=1:numel(snr)
		for noise_idx=1:numel(noise_type)
			cur_snr = snr(snr_idx);
			cur_noise = validation_noise_wav{noise_idx};
			for validation_idx=1:len_validation
				validation_noisy = gen_mix(...
								validation_wav{validation_idx},...
								cur_noise,...
								cur_snr);

				[tmp_feat, tmp_label] = get_training_data(...
									validation_wav{validation_idx},...
									validation_noisy,...
									'ARastaplpMfccGf',...
									win_len,...
									win_shift,...
									fs,...
									q,...
									c);
				validation_feat = [validation_feat; tmp_feat'];
				validation_label = [validation_label; tmp_label'];
			end
		end
	end
	if save_validation_data
		save([save_path, filesep, 'validation_data.mat'], 'validation_feat', 'validation_label');
	end
else
	load([save_path, filesep, 'validation_data.mat']);
end
% replace NaN with zeros
validation_label(find(isnan(validation_label))) = 0;

fprintf('...done\n');
validation_feat = single(validation_feat);

feat_len = size(validation_feat, 2);


start_step = 1;
% intialize the net
if isempty(checkpoint_path_net)
	net = gen_net(feat_len, ...
					win_len, ...
					adjacent_frame, ...
					hidden_layer_struct, ...
					useInputNormalization, ...
					isGPU);
else
	load(checkpoint_path_net);
	start_step = split(checkpoint_path_net, 'step');
	start_step = start_step{2};
	start_step = split(start_step, '_net');
	start_step = str2num(start_step{1});
end

% initialize the optimizer
if isempty(checkpoint_path_optimizer)
	optimizer = gen_training_optimizer(...
								initial_learn_rate, ...
								num_epoch, ...
								mini_batch_size, ...
								isGPU);
else
	load(checkpoint_path_optimizer);
end

% generate speech, noise and snr lists used for training an epoch
size_dataset = len_training * len_noise * numel(snr);
list_train_data = strings(size_dataset, 6);
fprintf('generating training data list...');
for snr_idx=1:numel(snr)
	for noise_idx=1: len_noise
		for speech_idx=1: len_training
			data_idx = speech_idx + (noise_idx-1) * len_training + (snr_idx-1) * len_noise * len_training;
			list_train_data(data_idx, 1) = string(speech_idx);
			list_train_data(data_idx, 2) = strcat(list_training(noise_idx).folder, filesep, list_training(noise_idx).name);
			list_train_data(data_idx, 3) = string(noise_idx);
			list_train_data(data_idx, 4) = strcat(list_noise(noise_idx).folder, filesep, list_noise(noise_idx).name);
			list_train_data(data_idx, 5) = string(snr_idx);
			list_train_data(data_idx, 6) = string(snr(snr_idx));
		end
	end
end

rng(random_seed);
shuffle_idx = randperm(size_dataset);
list_train_data = list_train_data(shuffle_idx, :);
fprintf('...done\n');
list_train_data = repmat(list_train_data, 2, 1);
single_batch_size = min(file_per_batch, size_dataset);

fprintf('-------------------------------------------------------\n')
fprintf('%-20s %-20s %-20s %s\n', 'Epoch', 'Train step', 'Cost', 'validtion mse');
total_train_steps = ceil(num_epoch * size_dataset / file_per_batch);
for training_iter=start_step:total_train_steps

	cur_epoch = floor(training_iter  * single_batch_size / size_dataset);
	% fprintf('Training. %d of %d epoch \n', cur_epoch, num_epoch);
	% fprintf('Training. %d of %d iterations \n', training_iter, total_train_steps);

	cur_idx = mod((training_iter-1) * single_batch_size, size_dataset) + 1; 
	batch_list_idx = cur_idx:cur_idx + single_batch_size-1;
	batch_list = list_train_data(batch_list_idx, :);

	% fprintf('obtaining training data...')
	if isempty(load_training_data)
		% read training files
        batch_feat = [];
		batch_label = [];
		batch_wav = cell(single_batch_size, 1);
		for training_idx=1:single_batch_size
			[batch_wav{training_idx}, fs0] = audioread(char(batch_list(training_idx, 2)));
			% resample to 16k Hz
			batch_wav{training_idx} = resample(batch_wav{training_idx}, 16e3, fs0);

			if ~isempty(wav_max_len)
				max_len = wav_max_len * 16e3;
				if length(batch_wav{training_idx}) > max_len
					batch_wav{training_idx} = batch_wav{training_idx}(1:max_len);
				else
					batch_wav{training_idx} = padarray(batch_wav{training_idx}, ...
													max_len - length(batch_wav{training_idx}),...
													'post');
%                     batch_wav{training_idx} = [batch_wav{training_idx};
%                                     1e-4*rand(max_len-length(validation_wav{validation_idx}), 1)];
				end
            end

            noise_idx = str2num(batch_list(training_idx, 3));
            snr_idx = str2num(batch_list(training_idx, 5));
            cur_snr = snr(snr_idx);
            cur_noise = noise_wav{noise_idx};
            
            training_noisy = gen_mix(...
                batch_wav{training_idx},...
                cur_noise,...
                cur_snr);
			[tmp_feat, tmp_label] = get_training_data(...
								batch_wav{training_idx}, ...
								training_noisy, ...
								'ARastaplpMfccGf',...
								win_len,...
								win_shift,...
								fs,...
								q,...
								c);
			batch_feat = [batch_feat; tmp_feat'];
			batch_label = [batch_label; tmp_label'];

		end

		batch_feat = single(batch_feat);
        batch_label(find(isnan(batch_label))) = 0;
		if save_training_data
			save([save_path, filesep, 'training_data.mat'], 'batch_feat', 'batch_label', 'list_train_data');
		end
	else
		load(load_training_data);
    end
   
    % fprintf('...done\n')
	if useInputNormalization
		[batch_feat_norm, net.norm_mu, net.norm_std] = mean_var_norm(batch_feat);
		batch_feat_win = win_buffer(batch_feat_norm, adjacent_frame);
	else
		batch_feat_win = win_buffer(batch_feat, adjacent_frame);
	end

	if useInputNormalization
		validation_feat_norm = mean_var_norm_testing(...
									validation_feat, ...
									net.norm_mu, ...
									net.norm_std);
		validation_feat_win = win_buffer(validation_feat_norm, adjacent_frame);
	else
		validation_feat_win = win_buffer(validation_feat, adjacent_frame);
	end
	[net, optimizer] = train_net(batch_feat_win, ...
									batch_label, ...
									validation_feat_win, ...
									validation_label, ...
									net, ...
									optimizer, ...
                                    cur_epoch, ...
                                    training_iter);

	if ~mod(training_iter, checkpoint_save_steps)
		save_checkpoint_path = [save_path, filesep, 'checkpoint'];
		if ~exist(save_checkpoint_path)
			mkdir(save_checkpoint_path);
		end
		save([save_checkpoint_path, filesep, 'checkpoint_step', ...
						num2str(training_iter), '_net.mat'], 'net');
		save([save_checkpoint_path, filesep, 'checkpoint_step', ...
						num2str(training_iter), '_optimizer.mat'], 'optimizer');
	end
end
fprintf('-------------------------------------------------------\n')

% generate test samples for evaluation
save_idx = 1;
fprintf('Testing. Total %d files...\n', numel(list_testing));

eval = [];
eval.pesq = zeros(numel(list_testing)*numel(snr)*numel(noise_type),3);
eval.ssnr = eval.pesq;
eval.stoi = eval.pesq;

eval.avg_pesq = zeros(numel(snr)*numel(noise_type), 3);
eval.avg_stoi = eval.avg_pesq;
eval.avg_ssnr = eval.avg_pesq;

for testing_item=1:numel(list_testing)
    fprintf('processing # %d file\n', testing_item);
	[testing_clean, fs0] = audioread([list_testing(testing_item).folder, filesep,...
							 char(list_testing(testing_item).name)]);
	testing_clean = resample(testing_clean, 16e3, fs0);

	if ~isempty(wav_max_len)
		max_len = wav_max_len * 16e3;
		if length(testing_clean) > max_len
			testing_clean = testing_clean(1:max_len);
		else
			testing_clean = padarray(testing_clean, ...
									max_len - length(testing_clean),...
									'post');
%             testing_clean = [testing_clean;
%                             zeros(max_len-testing_clean, 1)];
		end
	end

    testing_clean = testing_clean / max(abs(testing_clean));
	for snr_idx=1:numel(snr)
		for noise_idx=1:numel(noise_type)
			cur_snr = snr(snr_idx);
			cur_noise = test_noise_wav{noise_idx};

			testing_noisy = gen_mix(...
							testing_clean, ...
							cur_noise, ...
							cur_snr);
			[testing_feat, testing_label] = get_training_data(...
									testing_clean, ...
									testing_noisy, ...
									'ARastaplpMfccGf',...
									win_len,...
									win_shift,...
									fs,...
									q,...
									c);
			testing_label = testing_label';
			testing_feat = testing_feat';
            testing_label(find(isnan(testing_label))) = 0;

			if useInputNormalization
				testing_feat_norm = mean_var_norm_testing(...
													testing_feat, ...
													net.norm_mu, ...
													net.norm_std);
				testing_feat_win = win_buffer(testing_feat_norm, adjacent_frame);
			else
				testing_feat_win = win_buffer(testing_feat, adjacent_frame);
			end
			testing_predict = predict_from_net(net.layers, ...
												testing_feat_win, ...
												optimizer);
			testing_predict = gather(testing_predict');
			testing_estimated = wav_synthesis(testing_predict, ...
											testing_noisy, ...
											fs, ...
											win_len, ...
											win_shift, ...
											q, ...
											c);
			testing_ideal = wav_synthesis(testing_label', ...
										testing_noisy, ...
										fs, ...
										win_len, ...
										win_shift, ...
										q, ...
										c);

			min_len = min(min(length(testing_ideal), length(testing_ideal)), length(testing_noisy));
			testing_noisy = testing_noisy(1:min_len);
			testing_noisy = testing_noisy(:);
			testing_clean = testing_clean(1:min_len);
			testing_clean = testing_clean(:);
			testing_estimated = testing_estimated(1:min_len);
			testing_estimated = testing_estimated(:);
            testing_ideal = testing_ideal(1:min_len);
            testing_ideal = testing_ideal(:);

			eval.pesq(save_idx, 1) = pesq(testing_clean, testing_noisy, fs);
			eval.pesq(save_idx, 2) = pesq(testing_clean, testing_ideal, fs);
			eval.pesq(save_idx, 3) = pesq(testing_clean, testing_estimated, fs);

			eval.ssnr(save_idx, 1) = snrseg(testing_noisy, testing_clean, fs);
			eval.ssnr(save_idx, 2) = snrseg(testing_ideal, testing_clean, fs);
			eval.ssnr(save_idx, 3) = snrseg(testing_estimated, testing_clean, fs);

			eval.stoi(save_idx, 1) = stoi(testing_clean, testing_noisy, fs);
			eval.stoi(save_idx, 2) = stoi(testing_clean, testing_ideal, fs);
			eval.stoi(save_idx, 3) = stoi(testing_clean, testing_estimated, fs);

			avg_idx = noise_idx + numel(noise_type) * (snr_idx - 1);
			eval.avg_pesq(avg_idx, 1) = eval.avg_pesq(avg_idx, 1) + eval.pesq(save_idx, 1);
			eval.avg_pesq(avg_idx, 2) = eval.avg_pesq(avg_idx, 2) + eval.pesq(save_idx, 2);
			eval.avg_pesq(avg_idx, 3) = eval.avg_pesq(avg_idx, 3) + eval.pesq(save_idx, 3);

			eval.avg_ssnr(avg_idx, 1) = eval.avg_ssnr(avg_idx, 1) + eval.ssnr(save_idx, 1);
			eval.avg_ssnr(avg_idx, 2) = eval.avg_ssnr(avg_idx, 2) + eval.ssnr(save_idx, 2);
			eval.avg_ssnr(avg_idx, 3) = eval.avg_ssnr(avg_idx, 3) + eval.ssnr(save_idx, 3);

			eval.avg_stoi(avg_idx, 1) = eval.avg_stoi(avg_idx, 1) + eval.stoi(save_idx, 1);
			eval.avg_stoi(avg_idx, 2) = eval.avg_stoi(avg_idx, 2) + eval.stoi(save_idx, 2);
			eval.avg_stoi(avg_idx, 3) = eval.avg_stoi(avg_idx, 3) + eval.stoi(save_idx, 3);

			save_path_estimated = char(strcat(save_path, filesep,...
							 'testing', filesep,...
							 'estimated', filesep,...
							 num2str(cur_snr), filesep,...
							 noise_type(noise_idx)));
			save_path_clean = char(strcat(save_path, filesep, ...
							 'testing', filesep, ...
							 'clean', filesep, ...
							 num2str(cur_snr), filesep,...
							 noise_type(noise_idx)));
			save_path_ideal = char(strcat(save_path, filesep,...
							 'testing', filesep,...
							 'ideal', filesep,...
							 num2str(cur_snr), filesep,...
							 noise_type(noise_idx)));
			save_path_mix = char(strcat(save_path, filesep,...
							 'testing', filesep,...
							 'mix', filesep,...
							 num2str(cur_snr), filesep,...
							 noise_type(noise_idx)));
                         
			if ~exist(save_path_estimated)
				mkdir(save_path_estimated);
			end
			if ~exist(save_path_clean)
				mkdir(save_path_clean);
			end
			if ~exist(save_path_ideal)
				mkdir(save_path_ideal);
			end			
			if ~exist(save_path_mix)
				mkdir(save_path_mix);
            end			
            
			audiowrite( ...
				[save_path_estimated,filesep,...
				char(list_testing(testing_item).name)],...
				testing_estimated, ...
				fs);
			audiowrite( ...
				[save_path_clean,filesep, ...
				char(list_testing(testing_item).name)],...
				testing_clean, ...
				fs);
			audiowrite( ...
				[save_path_ideal,filesep,...
				char(list_testing(testing_item).name)],...
				testing_ideal, ...
				fs);
			testing_noisy = testing_noisy / max(abs(testing_noisy));
			audiowrite( ...
				[save_path_mix,filesep,...
				char(list_testing(testing_item).name)],...
				testing_noisy, ...
				fs);
			save_idx = save_idx + 1;
		end
	end
end
eval.avg_pesq = eval.avg_pesq ./ (numel(list_testing));
eval.avg_stoi = eval.avg_stoi ./ (numel(list_testing));
eval.avg_ssnr = eval.avg_ssnr ./ (numel(list_testing));

fprintf('Testing done\n');

save([save_path, filesep, 'testing', filesep, 'net.mat'], 'net');
save([save_path, filesep, 'testing', filesep, 'optimizer.mat'], 'optimizer');

fid=fopen([save_path, filesep, 'testing', filesep, 'info.txt'], 'w');
fprintf(fid, 'noise: %s \n', noise_type);
fprintf(fid, 'snr: %.0f \n', snr);
fprintf(fid, 'list of testing wav files: \n');
for testing_item=1:numel(list_testing)
	fprintf(fid, [char(list_testing(testing_item).name), '\n']);
end

% print the evaluation result
fprintf(fid, '\n');
print_eval_res(fid, eval, 'mix', noise_type, snr, list_testing, 0);
print_eval_res(fid, eval, 'ideal', noise_type, snr, list_testing, 0);
print_eval_res(fid, eval, 'estimated', noise_type, snr, list_testing, 0);

print_eval_res(fid, eval, 'mix', noise_type, snr, list_testing, 1);
print_eval_res(fid, eval, 'ideal', noise_type, snr, list_testing, 1);
print_eval_res(fid, eval, 'estimated', noise_type, snr, list_testing, 1);

fclose(fid);
diary off;