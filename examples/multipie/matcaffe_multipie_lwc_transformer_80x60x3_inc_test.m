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
use_gpu = true;
solver_file = 'examples/multipie/multipie_lwc_transformer_80x60x3_inc_solver.prototxt';
model_file = 'examples/multipie/multipie_lwc_transformer_80x60x3_inc_test.prototxt';
param = struct('base_lr', 0.000001, 'stepsize', 100000, 'snapshot', 100000, 'snapshot_prefix', 'examples/multipie/lwc_transformer');
make_solver_file_adagrad(solver_file, model_file, param);
matcaffe_init_multipie(true,solver_file);
batch_size = 100;

fid_test = fopen([path_to_results 'multipie_lwc_transformer_80x60x3_inc_test_errors.txt'],'w');
for n = 1000
    loss_test_image = 0;
    tic;
    transformer = load(sprintf([path_to_results 'multipie_lwc_transformer_80x60x3_model_inc_iter%04d.mat'], n));
    caffe('set_weights', transformer.weights);
    caffe('set_phase_test');
    for m = 1:length(out_idx_test),
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
        loss_image = loss_euclidean(recons_out, images_out);
        fprintf('reconstruction loss is %f\n', sqrt(2*loss_image));
        loss_test_image = loss_test_image + sqrt(2*loss_image);
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
        imwrite(cat(2, images_in, recons_out, images_out), sprintf('rotated_face_%02d.png',m), 'png');
        pause
end
    end      
%    figure(1); montage2(images_test(:,:,:,in_idx_test(1:16)),struct('hasChn',1));
%    figure(2); montage2(images_test(:,:,:,out_idx_test(1:16)),struct('hasChn',1));
%    figure(4); montage2(images_recons(:,:,:,1:16),struct('hasChn',1));
    loss_test_image = loss_test_image / m;
    fprintf(sprintf('LWCTransformer_80x60x3_inc -- testing losses are %f in %f seconds.\n', loss_test_image, toc));
    fprintf(fid_test, '%d %f\n', n, loss_test_image); 
    
end    
fclose(fid_test);

