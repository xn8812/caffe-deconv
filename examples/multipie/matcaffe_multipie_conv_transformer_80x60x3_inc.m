% add path
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
[h,w,c,num_train] = size(images_train);
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
% prepare train pairs
labels_train = mpplabel(train_idx);
ids_person = unique(labels_train); ids_person = ids_person(:)';
sessions_train = mpslabel(train_idx);
ids_sessions = unique(sessions_train); ids_sessions = ids_sessions(:)';
in_idx_train = [];
out_idx_train = [];
rclasses_train = [];
for cc = ids_person, 
    for ss = ids_sessions, 
        class_idx = find(labels_train==cc & sessions_train==ss); 
        if ~isempty(class_idx)
            theta_class = theta_train(class_idx);
            nn = length(class_idx);
            adj = repmat(theta_class(:),1,nn) - repmat(theta_class(:)',nn,1);
            [ii,jj] = find(abs(adj)<=15); 
            diff = adj(abs(adj)<=15); diff = diff(:)';
            in_idx = class_idx(ii); in_idx_train = [in_idx_train; in_idx(:)];
            out_idx = class_idx(jj); out_idx_train = [out_idx_train; out_idx(:)];
            rot = [diff==-15; diff==0; diff==15]; rclasses_train = [rclasses_train, rot];
        end
    end; 
end;
num_pairs = length(in_idx_train);

% init caffe network (spews logging info)
model_spec = 'multipie_conv_transformer_inc_v1';
use_gpu = true;
solver_file = sprintf('examples/multipie/%s_solver.prototxt', model_spec);
model_file = sprintf('examples/multipie/%s.prototxt', model_spec);
param = struct('base_lr', 0.0001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_multipie(true,solver_file,3);
batch_size = 100;

fid_train = fopen([path_to_results model_spec '_train_errors.txt'],'w');
for n = 1:1000
    loss_train_image = 0;
    tic;
    fprintf('%s -- processing the %dth iteration.\n', model_spec, n);
    rnd_idx = randperm(num_pairs);
    caffe('set_phase_train');
    for m = 1:floor(num_pairs/batch_size)
        %disp(m)
        batch_idx = rnd_idx((m-1)*batch_size+(1:batch_size));
        images_out = images_train(:,:,:,out_idx_train(batch_idx));
        images_out = reshape(single(images_out),[h,w,c,batch_size]);
        images_out = single(permute(images_out,[2,1,3,4]));
        rotations = rclasses_train(:,batch_idx);
        rotations = reshape(single(rotations),[1,1,3,batch_size]);
        images_in = images_train(:,:,:,in_idx_train(batch_idx));
        images_in = single(permute(images_in,[2,1,3,4]));
        results = caffe('forward', {images_in; rotations});
        %fprintf('Done with forward pass.\n');
        recons_out = results{1};
        loss_image = loss_euclidean(recons_out, images_out);
        delta_image = loss_euclidean_grad(recons_out, images_out);
        caffe('backward', {delta_image});
        caffe('update');
        %fprintf('Done with update\n');
        %save diff_cifar.mat d;
        loss_train_image = loss_train_image + loss_image/batch_size;
    end      
    loss_train_image = loss_train_image / m;
    fprintf(sprintf('%s -- training losses are %f in %f seconds.\n', model_spec, loss_train_image, toc));
    fprintf(fid_train, '%d %f\n', n, loss_train_image); 
    
    if mod(n,10)==0,
        weights = caffe('get_weights');
        save(sprintf([path_to_results '%s_model_iter%04d.mat'], model_spec, n), 'weights');
    end
end    
fclose(fid_train);

