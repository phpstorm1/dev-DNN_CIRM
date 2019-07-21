function combine = ARastaplpMfccGf(sig, win_len, win_shift)
    
    fs = 16000;

    % use a smaller hop size to improve the performance
    if nargin < 2
        win_len = 320;
        win_shift = 80;
    elseif nargin < 3
        win_shift = win_len / 4;
    end
    
    constant = 5*10e6; % used for engergy normalization for mixed signal

    sig = double(sig);
    sig = sig / max(abs(sig));

    c = sqrt(constant * length(sig)/sum(sig.^2));
    sig = sig*c;
    
    nframe = floor(length(sig)/win_shift);
    nc = 64;
    
    feat_stack = [];
    %fprintf('AMS...')
    ams_feat = get_amsfeature_chan_fast(sig, nc, fs, win_len, win_shift);

    %fprintf('RASTAPLP2D...')
    rastaplp_feat = rastaplp(sig, fs, 1, 12, win_len, win_shift);
    rastaplp_feat = [zeros(size(rastaplp_feat,1),1) rastaplp_feat];
    
    %fprintf('MODMFCC...\n')
    mfcc_feat = melfcc(sig, fs,'numcep',31,'nbands',nc,'wintime', win_len/fs, 'hoptime', win_shift/fs, 'maxfreq', 8000);
    mfcc_feat = [zeros(size(mfcc_feat,1),1) mfcc_feat];
     
    feat = cochleagram(gammatone(sig, nc, [50, 8000], fs), win_len, win_shift);
    %feat = feat(:,1:nFrame);
    gf_feat = single(feat.^(1/15));

    min_len = min(min(length(ams_feat),length(rastaplp_feat)), min(length(mfcc_feat),length(gf_feat)));
    ams_feat = ams_feat(:,1:min_len);
    rastaplp_feat = rastaplp_feat(:, 1:min_len);
    mfcc_feat = mfcc_feat(:, 1:min_len);
    gf_feat = gf_feat(:, 1:min_len);
    combine = [ams_feat; rastaplp_feat; mfcc_feat; gf_feat];  
    
    % add delta features
    combine = [combine; deltas(combine)];
end

