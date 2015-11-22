% testing chair transformer_3conv_inc_mask

% prepare oversampled input
path_to_data  = 'examples/chairs/data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
test_idx = ids>500;
images_test = images(:,:,:,test_idx);
[h,w,c,n] = size(images_test);
ids_test = ids(test_idx);
phi_test = phi(test_idx);
theta_test = theta(test_idx);
ids_types = unique(ids_test);
phi_types = unique(phi_test);
theta_types = unique(theta_test);

in_idx_test = [];
out_idx_test = [];
rclasses_test = [];
for cc = ids_types(:)',
  for tt = theta_types(:)',
    class_idx = find(ids_test==cc & theta_test==tt);
    phi_class = phi_test(class_idx);
    nn = length(class_idx);
    adj = repmat(phi_class(:),1,nn) - repmat(phi_class(:)',nn,1);
    [ii,jj] = find(abs(adj)<=12); 
    diff = adj(abs(adj)<=12); diff = diff(:)';
    in_idx = class_idx(ii); in_idx_test = [in_idx_test; in_idx(:)];
    out_idx = class_idx(jj); out_idx_test = [out_idx_test; out_idx(:)];
    rot = [diff<0; diff==0; diff>0]; rclasses_test = [rclasses_test, rot];
  end
end
num_pairs = length(in_idx_test);
num_pairs    
pause

model_def = '3conv_inc_mask_fixed';
% init caffe network (spews logging info)
use_gpu = true;
model_file = sprintf('examples/chairs/chairs_transformer_%s_test.prototxt', model_def);
solver_file = sprintf('examples/chairs/chairs_transformer_%s_solver.prototxt', model_def);
param = struct('base_lr', 0.0001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_chairs(use_gpu, solver_file);
batch_size = 1;

path_to_results = 'examples/chairs/results/';
if ~isdir(path_to_results), mkdir(path_to_results); end
for n = 20 
    if ~isdir(sprintf('chairs_rotated_mask_iter%d',n)), mkdir(sprintf('chairs_rotated_mask_iter%d',n)); end
    load(sprintf([path_to_results 'chairs_transformer_%s_model_iter%04d.mat'], model_def, n));
    caffe('set_weights', weights);
    loss_test_image = 0;
    tic;
    fprintf('Chair Transformer -- processing the %dth iteration.\n',n);
    caffe('set_phase_test'); 
    for cc = ids_types(:)',
	for tt = theta_types(:)',
            batch_idx = find(ids_test==cc & theta_test==tt);
	    batch_size = length(batch_idx);
	    images_batch = images_test(:,:,:,batch_idx);
	    phi_batch = phi_test(batch_idx);
	    [phi_batch, order] = sort(phi_batch, 'ascend');
	    images_in = images_batch(:,:,:,order);
	    images_out = images_in(:,:,:,[2:batch_size,1]);
            rotations = repmat([1;0;0],1,batch_size); 
            images_out = single(permute(images_out,[2,1,3,4]))/255;
            rotations = reshape(single(rotations),[1,1,3,batch_size]);
            images_in = single(permute(images_in,[2,1,3,4]))/255;
            results = caffe('forward', {images_in; rotations});
 	    recons_out = results{1};% + randn(1)*exp(0.5*log_var);
            masks_out = results{2};
            recons_out = recons_out .* repmat(masks_out, [1,1,3,1]);
      	    % save generated faces
	    for m = 1:batch_size,
		x = images_in(:,:,:,m);
		y_hat = recons_out(:,:,:,m);
		y = images_out(:,:,:,m);
	        err = norm(y(:)-y_hat(:));
	        tmp1 = uint8(255*squeeze(permute(1-y_hat,[2,1,3,4])));
	        tmp2 = uint8(255*squeeze(permute(1-x,[2,1,3,4])));
	        tmp3 = uint8(255*squeeze(permute(1-y,[2,1,3,4])));
	        imwrite(cat(2,tmp2,tmp1,tmp3),sprintf('chairs_rotated_mask_iter%d/inst%d_ele%d_azi%d.png',n, cc, tt, m),'png');
	        fprintf('reconstrction error is %f\n', err);
	     end
	     pause;
	end
    end      
    
end

