% add path
addpath(genpath('examples/multipie/'));
path_to_data  = 'examples/multipie/data/';
path_to_results = 'examples/multipie/results/';

% prepare input
[ mpfaces, mpplabel, mpilabel, mpvlabel, mpslabel, mpelabel ] = ...
    multipieLoadCropped([path_to_data 'nf_neu_' imsize '.mat']);
% using mvp setup
keepillum = mpilabel==0;
mpfaces = mpfaces(:,:,:,keepillum);
mpplabel = mpplabel(keepillum);
mpvlabel = mpvlabel(keepillum);
mpslabel = mpslabel(keepillum);

keepview = (mpvlabel==80)|(mpvlabel==130)|(mpvlabel==140)|(mpvlabel==51)...
    |(mpvlabel==50)|(mpvlabel==41)|(mpvlabel==190);

mpfaces = mpfaces(:,:,:,keepview);
mpfaces = im2double(mpfaces);
mpfaces = mpfaces - min(mpfaces(:));
mpfaces = mpfaces ./ max(mpfaces(:));
mpplabel = mpplabel(keepview);
mpvlabel = mpvlabel(keepview);
mpslabel = mpslabel(keepview);
[h,w,c,n] = size(mpfaces);
mean_face = zeros(h,w,c,'single');
for i=1:n, mean_face = mean_face*(i-1)/i + single(mpfaces(:,:,:,i))/i; end
[mpplabel,reord] = sort(mpplabel,'ascend');
mpfaces = mpfaces(:,:,:,reord);
mpvlabel = mpvlabel(reord);
mpslabel = mpslabel(reord);
save([path_to_data 'multipie_subset_id_pose_' imsize '.mat'],'mpfaces','mpplabel','mpvlabel','mpslabel','mean_face');