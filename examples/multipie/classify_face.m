% add path
addpath(genpath('examples/multipie/'));
path_to_data  = 'examples/multipie/data/';

% prepare input
if exist([path_to_data 'multipie_subset_id_pose_80x60.mat'],'file'),
    load([path_to_data 'multipie_subset_id_pose_80x60.mat']);
else
    prepare_data_id_pose;
end
% % relabeling
%ids_person = unique(mpplabel); 
% labels = zeros(size(mpplabel),'single');
% for i = 1:length(ids_person), labels(mpplabel==ids_person(i)) = i; end
labels = mpplabel;
ids_view = [80,130,140,51,50,41,190];
theta = -45:15:45;
views = zeros(2,length(mpvlabel),'single');
for i = 1:length(ids_view), 
    views(1,mpvlabel==ids_view(i)) = sind(theta(i));
    views(2,mpvlabel==ids_view(i)) = cosd(theta(i));
end


isLDA = false;
model_specs = 'multipie_conv_encoder';
use_gpu = true;
solver_file = sprintf('examples/multipie/%s_solver.prototxt', model_specs);
model_file = sprintf('examples/multipie/%s.prototxt', model_specs);
param = struct('base_lr', 0.00001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_multipie(true,solver_file);

% load pretested layer weights
model = caffe('get_weights');
numsteps = 1;
if numsteps == 1,
	model_pretrained = load('examples/multipie/results/multipie_conv_transformer_inc_model_iter0400.mat');
else
	model_pretrained = load(sprintf('examples/multipie/results/rnn_t%d_multipie_model_fixed_iter0010.mat', numsteps));
end
mapping = [1:5;1:5];
for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
caffe('set_weights', model);

% training LDA
if isLDA,
train_idx = labels<=200;
images_train = mpfaces(:,:,:,train_idx);
[h,w,c,num_train] = size(images_train);
views_train = views(:,train_idx);
theta_train = atand(views_train(1,:)./views_train(2,:)); theta_train = theta_train(:);
labels_train = labels(train_idx);
sessions_train = mpslabel(train_idx);
% extract features
feats_train = zeros(512,num_train,'single');
for m = 1:num_train
    images = images_train(:,:,:,m);
    images = reshape(single(images),[h,w,c,1]);
    images = permute(images,[2,1,3,4]);
    preds = caffe('forward', {images});
    feats_train(:,m) = squeeze(preds{1});
    if mod(m,100)==0, fprintf(sprintf('extract %d features\n', m)); end
end
feats_train = feats_train ./ repmat(sqrt(sum(feats_train.^2,1)),512,1);
W = LDA(labels_train, struct('PCARatio', 300), double(feats_train'));
end

test_idx = labels>200;
images_test = mpfaces(:,:,:,test_idx);
[h,w,c,num_test] = size(images_test);
views_test = views(:,test_idx);
theta_test = atand(views_test(1,:)./views_test(2,:)); theta_test = theta_test(:);
labels_test = labels(test_idx);
sessions_test = mpslabel(test_idx);
% prepare class index
% extract features
feats_test = zeros(512,num_test,'single');
for m = 1:num_test
    images = images_test(:,:,:,m);
    images = reshape(single(images),[h,w,c,1]);
    images = permute(images,[2,1,3,4]);
    preds = caffe('forward', {images});
    feats_test(:,m) = squeeze(preds{1});
    if mod(m,100)==0, fprintf(sprintf('extract %d features\n', m)); end
end
feats_test = feats_test ./ repmat(sqrt(sum(feats_test.^2,1)),512,1);
if isLDA,
	feats_test = W'*feats_test;
end


% gallery and probe split
diff_candidates = [15,30,45,60,75,90];
acc_test = cell(6,1);
for i = 1:6, acc_test{i} = []; end
for i = 1:7,

	select_view = theta(i);
	ind_gallery = zeros(size(labels_test));
	if 0,
	for cc = unique(labels_test)', 
	    session_cc = unique(sessions_test(labels_test==cc));
	    ss = min(session_cc);
	    ind = (labels_test==cc)&(sessions_test==ss)&(theta_test==select_view);
	    ind_gallery = ind_gallery + ind;
	end;
	end
	if 1,
	for cc = unique(labels_test)', 
	    session_cc = unique(sessions_test(labels_test==cc));
	    for ss = session_cc(:)'
	    ind = (labels_test==cc)&(sessions_test==ss)&(theta_test==select_view);
	    ind_gallery = ind_gallery + ind;
	    end
	end;
	end
	ind_gallery = ind_gallery > 0;
	ind_probe = (~ind_gallery);
	sum(ind_gallery)

	% gallery/probe split
	feats_gallery = feats_test(:,ind_gallery);
	feats_probe = feats_test(:,ind_probe);
	labels_gallery = labels_test(ind_gallery);
	labels_probe = labels_test(ind_probe);
	theta_gallery = theta_test(ind_gallery);
	theta_probe = theta_test(ind_probe);

	diff = abs(theta_probe - select_view);
	% similarity
	dist = EuDist2(feats_gallery', feats_probe', 1); 
	[~,matches] = min(dist,[],1);
	labels_pred = labels_gallery(matches);
	%%

	for j = 1:length(diff_candidates),
	    labels_pred_sub = labels_pred(diff==diff_candidates(j));
	    labels_probe_sub = labels_probe(diff==diff_candidates(j));
            if ~isempty(labels_probe_sub),
	        acc_test{j} = [acc_test{j} sum(labels_pred_sub==labels_probe_sub)/length(labels_probe_sub)];
	    end
	end
end
for j=1:6, mm(j) = mean(acc_test{j}); ss(j) = std(acc_test{j}); end 
mm(:)
ss(:)
