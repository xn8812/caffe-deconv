% add path
addpath(genpath('examples/multipie/'));
path_to_data  = 'examples/multipie/data/';

% prepare input
if exist([path_to_data 'multipie_subset_id_pose_48x48.mat'],'file'),
    load([path_to_data 'multipie_subset_id_pose_48x48.mat']);
else
    [ mpfaces, mpplabel, mpilabel, mpvlabel, mpslabel, mpelabel ] = ...
        multipieLoadCropped([path_to_data 'nf_neu_48x48.mat']);
    % using mvp setup
    keepillum = mpilabel==0;
    mpfaces = mpfaces(:,:,:,keepillum);
    mpplabel = mpplabel(keepillum);
    mpvlabel = mpvlabel(keepillum);

    keepview = (mpvlabel==80)|(mpvlabel==130)|(mpvlabel==140)|(mpvlabel==51)...
        |(mpvlabel==50)|(mpvlabel==41)|(mpvlabel==190);

    mpfaces = mpfaces(:,:,:,keepview);
    mpplabel = mpplabel(keepview);
    mpvlabel = mpvlabel(keepview);
    [h,w,c,n] = size(mpfaces);
    mean_face = zeros(h,w,c,'single');
    for i=1:n, mean_face = mean_face*(i-1)/i + single(mpfaces(:,:,:,i))/i; end
    save([path_to_data 'multipie_subset_id_pose_48x48.mat'],'mpfaces','mpplabel','mpvlabel','mean_face');
end
[mpplabel,reord] = sort(mpplabel,'ascend');
mpfaces = mpfaces(:,:,:,reord);
mpvlabel = mpvlabel(reord);
% relabeling
ids_person = unique(mpplabel); 
labels = zeros(1,length(mpplabel),'single');
for i = 1:length(ids_person), labels(mpplabel==ids_person(i)) = i; end
ids_view = [80,130,140,51,50,41,190];
theta = -45:15:45;
views = zeros(2,length(mpvlabel),'single');
for i = 1:length(ids_view), 
    views(1,mpvlabel==ids_view(i)) = sind(theta(i));
    views(2,mpvlabel==ids_view(i)) = cosd(theta(i));
end
% train/test split
images_test = mpfaces(:,:,:,labels>200);
images_test = mean(single(images_test),3);
views_test = views(:,labels>200);
labels_test = labels(labels>200);
[h,w,c,num_test] = size(images_test);
mean_face = mean(mean_face,3);

% init caffe network (spews logging info)
use_gpu = true;
matcaffe_init_multipie(use_gpu,'examples/multipie/multipie_classifier_solver_test.prototxt');

% start testing
path_to_results = 'examples/multipie/results/';
fid_id = fopen([path_to_results 'multipie_classifier_test_errors_id.txt'],'w');
for n = 100
    % load model
    load(sprintf([path_to_results 'multipie_classifier_model_iter%04d.mat'], n));
    caffe('set_weights', weights);
    % load([path_to_results 'lda_model_iter%04d.mat']);
    % extract features;
    feats_id = cell(1,num_test);
    theta = atand(views_test(1,:)./views_test(2,:));
    tic;
    caffe('set_phase_test');
    for m = 1:num_test
        %disp(m)
        images = images_test(:,:,:,m);
        images = reshape(single(images),[h,w,1,1]);
        images(:,:,:) = images(:,:,:) - mean_face;
        images = permute(images,[2,1,3,4])/255;
        preds = caffe('forward', {images});
        preds_class = preds{1};
        act = caffe('get_all_data');
        feats_id{m} = act(3).data(:);
    end
    feats_id = cell2mat(feats_id);
%     feats_id = W'*feats_id;
    feats_id = feats_id ./ repmat(sqrt(sum(feats_id.^2,1)),size(feats_id,1),1);
    % evaluate recognition
    ind_gallery = (theta==0);
    ind_probe = (theta~=0);
    feats_gallery = feats_id(:,ind_gallery);
    feats_probes = feats_id(:,ind_probe);
    labels_gallery = labels_test(ind_gallery);
    labels_probe = labels_test(ind_probe);
    theta_gallery = theta(theta==0);
    theta_probe = theta(theta~=0);
    sim = feats_gallery'*feats_probes;
    [~,matches] = max(sim,[],1);
    labels_pred = labels_gallery(matches);
    acc = [];
    for t = [-15,15,-30,30,-45,45],
        labels_pred_sub = labels_pred(theta_probe==t);
        labels_probe_sub = labels_probe(theta_probe==t);
        acc = [acc; sum(labels_pred_sub==labels_probe_sub)/length(labels_probe_sub)];
    end
    fprintf(fid_id, '%d %f %f %f %f %f %f %f\n', n, mean(acc), acc(1), acc(2), acc(3), acc(4), acc(5), acc(6));
end
fclose(fid_id);
acc
