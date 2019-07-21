function [ win_buf ] = win_buffer( feat, neighbor_frame )
%win_buffer Add additional frames to both sides, shape the frames to the shape
%	num_frames * feat_dim, where feat_dim = (2*neigbor_frame+1)*feat_len
%   Detailed explanation goes here

if neighbor_frame
	% feat: num_frames * feat_len
	[num_frames, feat_len] = size(feat);
	feat_rep = [repmat(feat(1,:), neighbor_frame, 1);
				feat;
				repmat(feat(end,:), neighbor_frame, 1)]';

	num_buffer_frames = 2*neighbor_frame + 1;
	win_buf = buffer(feat_rep(:), ...
						feat_len*num_buffer_frames, ...
						feat_len*num_buffer_frames-feat_len, ...
						'nodelay')';
else
	win_buf = feat;
	
end
