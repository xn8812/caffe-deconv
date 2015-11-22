
% prepare oversampled input
path_to_data  = 'examples/chairs/data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
test_idx = ((ids>500) & (ids<=520));
images_test = images(:,:,:,test_idx);
[h,w,c,numtests] = size(images_test);
ids_test = ids(test_idx);
phi_test = phi(test_idx);
theta_test = theta(test_idx);
ids_types = unique(ids_test);
phi_types = unique(phi_test);
theta_types = unique(theta_test);
batch_size = length(phi_types(:));


% init caffe network (spews logging info)
numsteps = 16;
model_specs = sprintf('rnn_t%d_finetuning_final',numsteps);
use_gpu = true;
model_file = sprintf('examples/chairs/%s.prototxt', model_specs);
solver_file = sprintf('examples/chairs/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.000001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_chairs(use_gpu, solver_file, 0);

% load pretested layer weights
if 1,
numsteps_prev = 16;
model = caffe('get_weights');
model_pretrained = load(sprintf('examples/chairs/results/rnn_t%d_finetuning_final_model_iter0050.mat',numsteps_prev));
mapping = [1:7;1:7]; 
for i = 1:numsteps, mapping = [mapping, [(8:18)+(i-1)*11; 8:18]]; end
for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
caffe('set_weights', model);
end

path_to_results = 'examples/chairs/results/';
if ~isdir(path_to_results), mkdir(path_to_results); end
fid_test = fopen([path_to_results sprintf('%s_model%d_test_errors.txt',model_specs,numsteps_prev)],'w');
for n = 50
    if 0,
    model = load(sprintf([path_to_results '%s_model_iter%04d.mat'], model_specs, n));
    caffe('set_weights', model.weights);
    end

    loss_test_image = zeros(1,16);
    tic;
    fprintf('%s -- processing the %dth iteration.\n', model_specs, n);
    m = length(ids_types(:))*length(theta_types(:))*length(phi_types(:))*2;
    for cc = [501:510]%ids_types(:)', % chair instance
	tic
	for tt = theta_types(:)', % elevation
            batch_idx = find(ids_test==cc & theta_test==tt);
	    images_batch = images_test(:,:,:,batch_idx);
            images_batch = single(permute(images_batch,[2,1,3,4]))/255;    
	    masks_batch = single(mean(images_batch,3)>0);
	    phi_batch = phi_test(batch_idx);
	    [phi_batch, order] = sort(phi_batch, 'ascend');
	    images_batch = images_batch(:,:,:,order);
	    masks_batch = masks_batch(:,:,:,order);
	    for ii = 1:batch_size, % azimuth
                for label = [1 3], % action
		    action = zeros(1,1,3,1,'single'); 
		    action(1,1,label,1) = 1;
		    input = cell(1+numsteps,1);
		    input{1} = images_batch(:,:,:,ii);
		    if label == 1, idx_rot = [ii+1:batch_size,1:ii]; end
		    if label == 3, idx_rot = [fliplr(1:ii-1),fliplr(ii:batch_size)]; end
		    images_out = images_batch(:,:,:,idx_rot);
		    images_out = reshape(images_out(:,:,:,1:numsteps), [w,h,3*numsteps,1]);
		    masks_out = masks_batch(:,:,:,idx_rot);
		    masks_out = reshape(masks_out(:,:,:,1:numsteps), [w,h,1*numsteps,1]); 
		    for ss = 1:numsteps, input{ss+1} = action; end
		    results = caffe('forward', input);
		    %fprintf('Done with forward pass.\n');
		    recons_image = results{1};
		    recons_mask = results{2};
if 0,
		    save([path_to_results sprintf('preds%d_inst%d_ele%d_azi%d_act%d_t%d.mat',numsteps_prev,cc,tt,ii,label,numsteps)], 'input', 'images_out', 'masks_out', 'recons_image', 'recons_mask');
end
		    for ss = 1:numsteps,
			recons = recons_image(:,:,(ss-1)*3+1:ss*3,1).*repmat(recons_mask(:,:,ss,1),[1,1,3,1]);
			loss = loss_euclidean(recons, images_out(:,:,(ss-1)*3+1:ss*3,1));
			loss_test_image(ss) = loss_test_image(ss) + loss;
		    end
 		end
            end
	end
	fprintf('--class %d in %f seconds\n',cc, toc);
    end      
    loss_test_image = loss_test_image ./ m;
    fprintf(sprintf('%s -- testing losses are %f for images in %f seconds.\n', model_specs, mean(loss_test_image), toc));
    for ss = 1:numsteps, fprintf(fid_test, '%d %f\n', ss, loss_test_image(ss)); end 
    
end
fclose(fid_test);

