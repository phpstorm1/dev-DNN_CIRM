noise_type = ["babble", "factory", "street", "restaurant"];
snr = -5:1:10;

win_len = 512;
win_shift = 256;
fs = 16e3;
q = 1;
c = 0.5;

% max length in seconds
wav_max_len = [];

validation_percentage = 0.1;
file_per_batch = 50;
total_train_steps = 3000;
validation_step = 100;
initial_learn_rate = 1e-3;
%learn_rate_decay_fac = 0.95;
useInputNormalization = 1;

mini_batch_size = 2048;
every_train_step = 20;
checkpoint_save_steps = 100;

checkpoint_path_net = './data/demo/checkpoint/checkpoint_step2000_net.mat';
checkpoint_path_optimizer = './data/demo/checkpoint/checkpoint_step2000_optimizer.mat';

save_training_data = 0;
load_training_data = '';

save_validation_data = 0;
load_validation_data = 1;

gpuDevice;
isGPU = gpuDeviceCount;

hidden_layer_struct = [1024, 1024, 1024];

speech_path = '/datasets/TIMIT/train_1';
test_path = '/datasets/TIMIT/test_1';
noise_path = '/datasets/noise';
save_path = './data/demo';

read_from_list = 0;
use_all_noise = 1;
train_list_path = './list/training_list731.txt';
test_list_path = './list/testing_list87.txt';
% train_list_path = '.\\list\\list300.txt';
% test_list_path = '.\\list\\list80.txt';

adjacent_frame = 2;

