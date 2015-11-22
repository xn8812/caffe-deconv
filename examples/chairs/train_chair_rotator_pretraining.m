% training chair transformer_disentangle_inc

% prepare oversampled input
path_to_data  = 'examples/chairs/data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
train_idx = ids<=500;
test_idx = ids>500;
images_train = images(:,:,:,train_idx);
[h,w,c,numtrains] = size(images_train);
ids_train = ids(train_idx);
phi_train = phi(train_idx);
theta_train = theta(train_idx);
ids_types = unique(ids_train);
phi_types = unique(phi_train);
theta_types = unique(theta_train);


% init caffe network (spews logging info)
numsteps = 2;
model_specs = 'rnn_t2_pretrained';
use_gpu = true;
model_file = sprintf('examples/chairs/%s.prototxt', model_specs);
solver_file = sprintf('examples/chairs/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.00001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_chairs(use_gpu, solver_file);
batch_size = 31;

% load pretrained layer weights
model = caffe('get_weights');
model_pretrain = load('examples/chairs/results/chairs_transformer_3conv_inc_mask_model_iter0500.mat');
mapping = [1:7;1:7]; mapping = [mapping, [8:18; 8:18]]; mapping = [mapping, [19:29; 8:18]];
for i = 1:size(mapping,2),
    model(mapping(1,i)).weights = model_pretrain.weights(mapping(2,i)).weights;
end
caffe('set_weights', model);

path_to_results = 'examples/chairs/results/';
if ~isdir(path_to_results), mkdir(path_to_results); end
fid_train = fopen([path_to_results sprintf('%s_train_errors.txt',model_specs)],'w');
for n = 1:1000
    loss_train_image = 0;
    tic;
    fprintf('%s -- processing the %dth iteration.\n', model_specs, n);
    m = length(ids_types(:))*length(theta_types(:));
    for cc = ids_types(:)',
	for tt = theta_types(:)',
            batch_idx = find(ids_train==cc & theta_train==tt);
	    batch_size = length(batch_idx);
	    images_batch = images_train(:,:,:,batch_idx);
            images_batch = single(permute(images_batch,[2,1,3,4]))/255;    
	    masks_batch = single(mean(images_batch,3)>0);
	    phi_batch = phi_train(batch_idx);
	    [phi_batch, order] = sort(phi_batch, 'ascend');
	    images_batch = images_batch(:,:,:,order);
	    masks_batch = masks_batch(:,:,:,order);
	    action = zeros(1,1,3,1); action(1,1,1,1) = 1;
	    rotations = single(repmat(action, [1,1,1,batch_size]));
	    input = cell(1+numsteps,1);
	    input{1} = images_batch; input{2} = rotations;
	    images_out = circshift(images_batch, 1, 4);
	    masks_out = circshift(masks_batch, 1, 4);
            for ss = 2:numsteps,
		images_out = cat(3, images_out, circshift(images_batch, ss, 4));
                masks_out = cat(3, masks_out, circshift(masks_batch, ss, 4));
		input{ss+1} = rotations;
            end
            results = caffe('forward', input);
            %fprintf('Done with forward pass.\n');
            recons_image = results{1};
	    recons_mask = results{2};
            loss_image = loss_euclidean(recons_image, images_out);
            loss_mask = loss_euclidean(recons_mask, masks_out);
            delta_image = loss_euclidean_grad(recons_image, images_out);
            delta_mask = loss_euclidean_grad(recons_mask, masks_out);
            caffe('backward', {delta_image; delta_mask});
            caffe('update');
            %fprintf('Done with update\n');
            loss_train_image = loss_train_image + (10*loss_image+loss_mask)/batch_size;
	end
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

