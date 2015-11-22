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
test_idx = mpplabel>200;
images_test = mpfaces(:,:,:,test_idx);
[h,w,c,num_test] = size(images_test);
% prepare train views
ids_view = [80,130,140,51,50,41,190];
theta = -45:15:45;
views = zeros(2,length(mpvlabel),'single');
for i = 1:length(ids_view), 
    views(1,mpvlabel==ids_view(i)) = sind(theta(i));
    views(2,mpvlabel==ids_view(i)) = cosd(theta(i));
end
views_test = views(:,test_idx);
theta_test = atand(views_test(1,:)./views_test(2,:));
% prepare train pairs
labels_test = mpplabel(test_idx);
ids_person = unique(labels_test); ids_person = ids_person(:)';
sessions_test = mpslabel(test_idx);
ids_sessions = unique(sessions_test); ids_sessions = ids_sessions(:)';
in_idx_test = [];
out_idx_test = [];
rclasses_test = [];
for cc = ids_person, 
    for ss = ids_sessions, 
        class_idx = find(labels_test==cc & sessions_test==ss); 
        if ~isempty(class_idx)
            theta_class = theta_test(class_idx);
            nn = length(class_idx);
            adj = repmat(theta_class(:),1,nn) - repmat(theta_class(:)',nn,1);
            [ii,jj] = find(abs(adj)<=15); 
            diff = adj(abs(adj)<=15); diff = diff(:)';
            in_idx = class_idx(ii); in_idx_test = [in_idx_test; in_idx(:)];
            out_idx = class_idx(jj); out_idx_test = [out_idx_test; out_idx(:)];
            rot = [diff==-15; diff==0; diff==15]; rclasses_test = [rclasses_test, rot];
        end
    end; 
end;
num_pairs = length(in_idx_test);

% init caffe network (spews logging info)
model_specs = 'multipie_conv_transformer_inc';
use_gpu = true;
solver_file = sprintf('examples/multipie/%s_solver.prototxt', model_specs);
model_file = sprintf('examples/multipie/%s_test.prototxt', model_specs);
param = struct('base_lr', 0.0001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_multipie(true,solver_file);
batch_size = 100;

mkdir('conv_transformer');
fid_test = fopen([path_to_results model_specs '_test_errors.txt'],'w');
for n = 400
    loss_test_image = 0;
    tic;
    transformer = load(sprintf([path_to_results model_specs '_model_iter%04d.mat'], n));
    caffe('set_weights', transformer.weights);
    caffe('set_phase_test');
    images_recons = zeros(h,w,c,length(out_idx_test));
    for m = 1:100
        %disp(m)
        images_out = images_test(:,:,:,out_idx_test(m));
        images_out = reshape(single(images_out),[h,w,c,1]);
        images_out = single(permute(images_out,[2,1,3,4]));
        rotations = rclasses_test(:,m);
        rotations = reshape(single(rotations),[1,1,3,1]);
        images_in = images_test(:,:,:,in_idx_test(m));
        images_in = single(permute(images_in,[2,1,3,4]));
        results = caffe('forward', {images_in; rotations});
        %fprintf('Done with forward pass.\n');
        recons_out = results{1};
        fprintf('generate novel images\n');
        loss_image = loss_euclidean(recons_out, images_out);
        loss_test_image = loss_test_image + loss_image;
        fprintf('loss is %f\n', loss_image);
        tmp = squeeze(permute(recons_out,[2,1,3,4]));
        tmp = tmp - min(tmp(:)); tmp = tmp ./ max(tmp(:));
        recons_out = tmp;
if 1,
        %figure(5); subplot(1,3,1); imshow(squeeze(permute(images_in,[2,1,3,4]))); title('before');
        tmp = squeeze(permute(images_in,[2,1,3,4]));
        tmp = tmp - min(tmp(:)); tmp = tmp ./ max(tmp(:)); 
        images_in = tmp;
        rot = squeeze(rotations);
        %subplot(1,3,2); imshow(squeeze(permute(images_out,[2,1,3,4]))); title(sprintf('%d %d %d',rot(1),rot(2),rot(3)));
	tmp = squeeze(permute(images_out,[2,1,3,4]));
	tmp = tmp - min(tmp(:)); tmp = tmp ./ max(tmp(:)); 
        images_out = tmp;
        %subplot(1,3,3); imshow(squeeze(permute(recons_out,[2,1,3,4]))); title(sprintf('%d %d %d',rot(1),rot(2),rot(3)));
        imwrite(cat(2, images_in, recons_out, images_out), sprintf('conv_transformer/rotated_face_%02d.png',m), 'png');
end
    end      
    loss_test_image = loss_test_image / m;
    fprintf(sprintf('%s -- testing losses are %f in %f seconds.\n', model_specs, loss_test_image, toc));
    fprintf(fid_test, '%d %f\n', n, loss_test_image); 
    
end    
fclose(fid_test);

