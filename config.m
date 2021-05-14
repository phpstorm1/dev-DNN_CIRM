noise_type = ["babble"];
snr = 0:1;

win_len = 512;
win_shift = 256;
fs = 16e3;
q = 1;
c = 0.5;

% max length in seconds
wav_max_len = [];


num_epoch = 10;
random_seed = 32;
file_per_batch = 20;
initial_learn_rate = 1e-3;
%learn_rate_decay_fac = 0.95;
useInputNormalization = 1;
mini_batch_size = 1024;

checkpoint_save_steps = 1;
checkpoint_path_net = '';
checkpoint_path_optimizer = '';

save_training_data = 1;
load_training_data = '';

validation_percentage = 15;
save_validation_data = 1;
load_validation_data = 0;

gpuDevice;
isGPU = gpuDeviceCount;

hidden_layer_struct = [512, 512, 512];

speech_path = './data/train';
test_path = './data/test';
noise_path = './data/noise';
save_path = './data/demo';

read_from_list = 0;
use_all_noise = 1;
train_list_path = './list/training_list731.txt';
test_list_path = './list/testing_list87.txt';
% train_list_path = '.\\list\\list300.txt';
% test_list_path = '.\\list\\list80.txt';

adjacent_frame = 2;

