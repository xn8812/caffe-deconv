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
images_train = mpfaces(:,:,:,labels<=200);
images_train = mean(single(images_train),3);
views_train = views(:,labels<=200);
labels_train = labels(labels<=200);
classes_train = sparse(double(labels_train(:)),double(1:length(labels_train)),...
    ones(length(labels_train),1,'double'),200,length(labels_train));
classes_train = single(full(classes_train));
[h,w,c,num_train] = size(images_train);
dim2 = size(classes_train,1);
dim3 = size(views_train,1);
mean_face = mean(mean_face,3);


% init caffe network (spews logging info)
use_gpu = true;
matcaffe_init_multipie(use_gpu,'examples/multipie/multipie_classifier_solver.prototxt');
batch_size = 100;

% start sgd training
path_to_results = 'examples/multipie/results/';
fid = fopen([path_to_results 'multipie_classifier_errors.txt'],'w');
for n = 1:100
    loss_train_class = 0;
    error_train_class = 0;
    tic;
    fprintf('Classifier -- processing the %dth iteration.\n',n);
    rnd_idx = randperm(num_train);
    caffe('set_phase_train');
    for m = 1:floor(num_train/batch_size)
        %disp(m)
        batch_idx = rnd_idx((m-1)*batch_size+(1:batch_size));
        images = images_train(:,:,:,batch_idx);
        images = reshape(single(images),[h,w,1,batch_size]);
        for t=1:batch_size, images(:,:,:,t) = images(:,:,:,t) - mean_face; end
        images = permute(images,[2,1,3,4]);
        classes = classes_train(:,batch_idx);
        classes = reshape(single(classes),[1,1,dim2,batch_size]);
        views = views_train(:,batch_idx);
        views = reshape(single(views),[1,1,dim3,batch_size]);
        preds = caffe('forward', {images});
        preds_class = preds{1};
        [loss_class,delta_class] = loss_crossentropy_paired_sigmoid_grad(preds_class, classes);
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
        save(sprintf([path_to_results 'multipie_classifier_model_iter%04d.mat'], n), 'weights');
    end
    
    fprintf(fid, '%d %f\n', n, error_train_class);
end
fclose(fid);
