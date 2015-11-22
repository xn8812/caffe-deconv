% add path
if 0,
addpath(genpath('examples/multipie/'));
path_to_data  = 'examples/multipie/data/';
path_to_results = 'examples/multipie/results/';

% prepare input
imsize = '80x60';
if exist([path_to_data 'multipie_subset_id_pose_' imsize '.mat'],'file'),
    load([path_to_data 'multipie_subset_id_pose_' imsize '.mat']);
else
    prepare_data_id_pose;
end

% prepare training images
train_idx = mpplabel<=200;
images_train = mpfaces(:,:,:,train_idx);
[h,w,c,numtrains] = size(images_train);
% prepare train views
ids_view = [80,130,140,51,50,41,190];
theta = -45:15:45;
views = zeros(2,length(mpvlabel),'single');
for i = 1:length(ids_view), 
    views(1,mpvlabel==ids_view(i)) = sind(theta(i));
    views(2,mpvlabel==ids_view(i)) = cosd(theta(i));
end
views_train = views(:,train_idx);
theta_train = atand(views_train(1,:)./views_train(2,:));
theta_types = unique(theta_train); theta_types = theta_types(:)';
ids_train = mpplabel(train_idx);
ids_types = unique(ids_train); ids_types = ids_types(:)';
sessions_train = mpslabel(train_idx);
session_types = unique(sessions_train); session_types = session_types(:)';
classes_train = sparse(double(ids_train(:)),double(1:length(ids_train)),...
ones(length(ids_train),1,'double'),length(ids_types),length(ids_train));
classes_train = single(full(classes_train));
end

% add path
addpath(genpath('examples/multipie/'));
path_to_data  = 'examples/multipie/data/';

% prepare input
if exist([path_to_data 'multipie_subset_id_pose.mat'],'file'),
    load([path_to_data 'multipie_subset_id_pose.mat']);
else
    [ mpfaces, mpplabel, mpilabel, mpvlabel, mpslabel, mpelabel ] = ...
        multipieLoadCropped([path_to_data 'nf_neu_80x60.mat']);
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
    save([path_to_data 'multipie_subset_id_pose.mat'],'mpfaces','mpplabel','mpvlabel','mean_face');
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
train_idx = labels<=200;
images_train = mpfaces(:,:,:,train_idx);
[h,w,c,num_train] = size(images_train);
views_train = views(:,train_idx);
theta_train = atand(views_train(1,:)./views_train(2,:)); theta_train = theta_train(:);
labels_train = labels(train_idx); labels_train = labels_train(:)';
sessions_train = mpslabel(train_idx); sessions_train = sessions_train(:)';

%
batch_size = 100;
model_specs = 'cnn_multipie';
use_gpu = true;
solver_file = sprintf('examples/multipie/%s_solver.prototxt', model_specs);
model_file = sprintf('examples/multipie/%s.prototxt', model_specs);
param = struct('base_lr', 0.0001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 0);
make_solver_file(solver_file, model_file, param);
matcaffe_init_multipie(true,solver_file,2);

numsteps = 0;
if numsteps > 0,
% load pretrained layer weights
model = caffe('get_weights');
if numsteps == 2,
	model_pretrained = load('examples/multipie/results/multipie_conv_transformer_inc_model_iter0400.mat');
else
	model_pretrained = load(sprintf('examples/multipie/results/rnn_t%d_multipie_model_fixed_iter0010.mat', numsteps-2));
end
mapping = [1:5;1:5];
for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
caffe('set_weights', model);

% start sgd training
fid = fopen([path_to_results model_specs '_classifier_errors.txt'],'w');
for n = 1:100
    loss_train_class = 0;
    error_train_class = 0;
    tic;
    fprintf('Classifier -- processing the %dth iteration.\n',n);
    rnd_idx = randperm(numtrains);
    caffe('set_phase_train');
    for m = 1:floor(numtrains/batch_size)
        %disp(m)
        batch_idx = rnd_idx((m-1)*batch_size+(1:batch_size));
        images = images_train(:,:,:,batch_idx);
        images = reshape(single(images),[h,w,c,batch_size]);
        for t=1:batch_size, images(:,:,:,t) = images(:,:,:,t) - mean_face; end
        images = permute(images(:,:,[3,2,1],:),[2,1,3,4])/255;
        classes = classes_train(:,batch_idx);
        classes = reshape(single(classes),[1,1,size(classes,1),batch_size]);
        preds = caffe('forward', {images});
        preds_class = preds{1};
        [loss_class,delta_class] = loss_crossentropy_paired_softmax_grad(preds_class, classes);
        [~,labels_pred] = max(squeeze(act_sigmoid(preds_class)),[],1);
        [~,labels_gt] = max(squeeze(classes),[],1);
        error_class = sum(labels_pred~=labels_gt);
        caffe('backward', {delta_class});
        caffe('update');
        loss_train_class = loss_train_class + loss_class;
        error_train_class = error_train_class + error_class;
    end      
    loss_train_class = loss_train_class / m;
    error_train_class = error_train_class / m;
    fprintf(sprintf('Classifier -- training losses are %f and training errors %f.\n', ...
        loss_train_class, error_train_class));
    
    if mod(n,10)==0,
        weights = caffe('get_weights');
        save(sprintf([path_to_results model_specs '_t%d_model_iter%04d.mat'], numsteps, n), 'weights');
    end
    
    fprintf(fid, '%d %f\n', n, error_train_class);
end
fclose(fid);
