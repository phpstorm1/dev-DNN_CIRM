function AMS_points = get_amsfeature_chan_fast(sub_gf,nChan,fs,win_len, win_shift)
	if nargin < 3
		fs = 16e3;
		win_len = 320;
		win_shift = win_len / 4;
    elseif nargin < 4;
		win_len = 320;
		win_shift = win_len / 4;
    elseif nargin < 5;
		win_shift = win_len / 4;
	end

sub_gf = reshape(sub_gf, 1, length(sub_gf));
nFrame = floor(length(sub_gf)/win_shift)-1;

ns_ams = extract_AMS_perChan(sub_gf, nChan, fs, win_len, win_shift);

AMS_points = ns_ams';
AMS_points = AMS_points(1:nFrame,:);

AMS_points = AMS_points.';
AMS_points = [zeros(size(AMS_points,1),1) AMS_points];
