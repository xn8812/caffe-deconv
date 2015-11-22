% training chair transformer_disentangle_inc

% prepare oversampled input
path_to_data  = 'examples/chairs/data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
train_idx = ids<=500;
test_idx = ids>500;
images_train = images(:,:,:,train_idx);
[h,w,c,n] = size(images_train);
ids_train = ids(train_idx);
phi_train = phi(train_idx);
theta_train = theta(train_idx);
ids_types = unique(ids_train);
phi_types = unique(phi_train);
theta_types = unique(theta_train);

in_idx_train = [];
out_idx_train = [];
rclasses_train = [];
for cc = ids_types(:)',
  for tt = theta_types(:)',
    class_idx = find(ids_train==cc & theta_train==tt);
    phi_class = phi_train(class_idx);
    nn = length(class_idx);
    adj = repmat(phi_class(:),1,nn) - repmat(phi_class(:)',nn,1);
    [ii,jj] = find(abs(adj)<=12 & abs(adj)~=0); 
    diff = adj(abs(adj)<=12 & abs(adj)~=0); diff = diff(:)';
    in_idx = class_idx(ii); in_idx_train = [in_idx_train; in_idx(:)];
    out_idx = class_idx(jj); out_idx_train = [out_idx_train; out_idx(:)];
    rot = [diff<0; diff>0]; rclasses_train = [rclasses_train, rot];
  end
end
num_pairs = length(in_idx_train);
num_pairs    

% init caffe network (spews logging info)
model_specs = 'rnn_t2';
use_gpu = true;
model_file = sprintf('examples/chairs/%s.prototxt', model_specs);
solver_file = sprintf('examples/chairs/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.0001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_chairs(use_gpu, solver_file);
batch_size = 100;

path_to_results = 'examples/chairs/results/';
if ~isdir(path_to_results), mkdir(path_to_results); end
fid_train = fopen([path_to_results sprintf('%s_train_errors.txt',model_specs)],'w');
for n = 1:1000
    loss_train_image = 0;
    tic;
    fprintf('%s -- processing the %dth iteration.\n', model_specs, n);
    rnd_idx = randperm(num_pairs);
    caffe('set_phase_train');
    for m = 1:floor(num_pairs/batch_size)
        %disp(m)
        batch_idx = rnd_idx((m-1)*batch_size+(1:batch_size));
        rotations = rclasses_train(:,batch_idx);
        rotations = reshape(single(rotations),[1,1,2,batch_size]);
        images_in = images_train(:,:,:,in_idx_train(batch_idx));
        images_in = single(permute(images_in,[2,1,3,4]))/255;
	masks_in = single(mean(images_in,3)>0);
        results = caffe('forward', {images_in; rotations});
        %fprintf('Done with forward pass.\n');
        images_out = images_train(:,:,:,out_idx_train(batch_idx));
        images_out = reshape(single(images_out),[h,w,3,batch_size]);
        images_out = single(permute(images_out,[2,1,3,4]))/255;
 	masks_out = single(mean(images_out,3)>0);
	images_out = cat(3,images_in,images_out);
	masks_out = cat(3,masks_in,masks_out);
        recons_image = results{1};
	recons_mask = results{2};
        loss_image = loss_euclidean(recons_image, images_out);
        loss_mask = loss_euclidean(recons_mask, masks_out);
        delta_image = loss_euclidean_grad(recons_image, images_out);
        delta_mask = loss_euclidean_grad(recons_mask, masks_out);
        caffe('backward', {delta_image; delta_mask});
        caffe('update');
        %fprintf('Done with update\n');
        %save diff_cifar.mat d;
        loss_train_image = loss_train_image + (10*loss_image+loss_mask)/batch_size;
    end      
    loss_train_image = loss_train_image / m;
    fprintf(sprintf('%s -- training losses are %f for images in %f seconds.\n', model_specs, loss_train_image, toc));
    fprintf(fid_train, '%d %f\n', n, loss_train_image); 
    
    if mod(n,10)==0,
        weights = caffe('get_weights');
        save(sprintf([path_to_results '%s_model_iter%04d.mat'], model_specs, n), 'weights');
    end
    
end
fclose(fid_train);

