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
batch_size = length(theta_types);

% init caffe network (spews logging info)
numsteps = 6;
model_specs = sprintf('rnn_t%d_multipie', numsteps);
use_gpu = true;
solver_file = sprintf('examples/multipie/%s_solver.prototxt', model_specs);
model_file = sprintf('examples/multipie/%s.prototxt', model_specs);
param = struct('base_lr', 0.00001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_multipie(true,solver_file);

% load pretrained layer weights
model = caffe('get_weights');
if numsteps == 2,
	model_pretrained = load('examples/multipie/results/multipie_conv_transformer_inc_model_iter0400.mat');
else
	model_pretrained = load(sprintf('examples/multipie/results/rnn_t%d_multipie_model_fixed_iter0010.mat', numsteps-2));
end
mapping = [1:5;1:5];
for i = 1:numsteps, mapping = [mapping, [(6:10)+(i-1)*5; 6:10]]; end
for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
caffe('set_weights', model);

fid_train = fopen([path_to_results model_specs '_train_errors.txt'],'w');
for n = 1:10
    loss_train_image = 0;
    tic;
    fprintf('%s -- processing the %dth iteration.\n', model_specs, n);
    m = 1;
    for cc = ids_types(:)', % chair instance
	for tt = session_types(:)', % elevation
            batch_idx = find(ids_train==cc & sessions_train==tt);
	    if isempty(batch_idx), continue; end
	    images_batch = images_train(:,:,:,batch_idx);
            images_batch = single(permute(images_batch,[2,1,3,4]));    
	    theta_batch = theta_train(batch_idx);
	    [theta_batch, order] = sort(theta_batch, 'ascend');
	    images_batch = images_batch(:,:,:,order);
	    images_batch = cat(4, images_batch, images_batch(:,:,:,batch_size-1:-1:1));
            labels = [ones(1,batch_size), 3*ones(1,batch_size-1)];	
	    for ii = 1:length(labels)-numsteps,
		    action = zeros(1,1,3,1,'single'); 
		    input = cell(1+numsteps,1);
		    input{1} = images_batch(:,:,:,ii);
		    idx_rot = [ii+1:ii+numsteps];
		    images_out = images_batch(:,:,:,idx_rot);
		    images_out = reshape(images_out, [w,h,3*numsteps,1]);
		    for ss = 1:numsteps,
		        action(1,1,labels(ii+ss),1) = 1;
			input{ss+1} = action;
		    end
		    results = caffe('forward', input);
		    %fprintf('Done with forward pass.\n');
		    recons_image = results{1};
		    loss_image = loss_euclidean(recons_image, images_out);
		    delta_image = loss_euclidean_grad(recons_image, images_out);
		    %fprintf('Done with delta\n');
		    caffe('backward', {delta_image});
		    caffe('update');
		    %fprintf('Done with update\n');
		    loss_train_image = loss_train_image + loss_image/numsteps;
		    m = m + 1;
            end
	end
    end      
    loss_train_image = loss_train_image / m;
    fprintf(sprintf('%s -- training losses are %f for images in %f seconds.\n', model_specs, loss_train_image, toc));
    fprintf(fid_train, '%d %f\n', n, loss_train_image); 
    
    if mod(n,1)==0,
        weights = caffe('get_weights');
        save(sprintf([path_to_results '%s_model_fixed_iter%04d.mat'], model_specs, n), 'weights');
    end
    
end
fclose(fid_train);

