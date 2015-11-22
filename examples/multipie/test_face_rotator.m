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

% prepare testing images
test_idx = mpplabel>200;
images_test = mpfaces(:,:,:,test_idx);
[h,w,c,numtests] = size(images_test);
% prepare test views
ids_view = [80,130,140,51,50,41,190];
theta = -45:15:45;
views = zeros(2,length(mpvlabel),'single');
for i = 1:length(ids_view), 
    views(1,mpvlabel==ids_view(i)) = sind(theta(i));
    views(2,mpvlabel==ids_view(i)) = cosd(theta(i));
end
views_test = views(:,test_idx);
theta_test = atand(views_test(1,:)./views_test(2,:));
theta_types = unique(theta_test); theta_types = theta_types(:)';
ids_test = mpplabel(test_idx);
ids_types = unique(ids_test); ids_types = ids_types(:)';
sessions_test = mpslabel(test_idx);
session_types = unique(sessions_test); session_types = session_types(:)';
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

% load pretested layer weights
for numsteps_prev = 1,

	model = caffe('get_weights');
	if numsteps_prev == 1,
		model_pretrained = load('examples/multipie/results/multipie_conv_transformer_inc_model_iter0400.mat');
	else
		model_pretrained = load(sprintf('examples/multipie/results/rnn_t%d_multipie_model_fixed_iter0010.mat', numsteps_prev));
	end
	mapping = [1:5;1:5];
	for i = 1:numsteps, mapping = [mapping, [(6:10)+(i-1)*5; 6:10]]; end
	for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
	caffe('set_weights', model);

	path_to_results = 'examples/multipie/results/';
if 1,
	fid_test = fopen([path_to_results sprintf('%s_model%d_test_errors.txt',model_specs,numsteps_prev)],'w');
end

    loss_test_image = zeros(1,6);
    tic;
    m = 1;
    for cc = ids_types(:)', % face instance
	for tt = session_types(:)', % session
            batch_idx = find(ids_test==cc & sessions_test==tt);
	    if isempty(batch_idx), continue; end
	    images_batch = images_test(:,:,:,batch_idx);
            images_batch = single(permute(images_batch,[2,1,3,4]));    
	    theta_batch = theta_test(batch_idx);
	    [theta_batch, order] = sort(theta_batch, 'ascend');
	    images_batch = images_batch(:,:,:,order);
	    images_batch = cat(4, images_batch, images_batch(:,:,:,batch_size-1:-1:1));
            labels = [ones(1,batch_size), 3*ones(1,batch_size-1)];	
	    for ii = [7]%:length(labels)-numsteps,
		    input = cell(1+numsteps,1);
		    input{1} = images_batch(:,:,:,ii);
		    idx_rot = [ii+1:ii+numsteps];
		    images_out = images_batch(:,:,:,idx_rot);
		    images_out = reshape(images_out, [w,h,3*numsteps,1]);
		    for ss = 1:numsteps,
		        action = zeros(1,1,3,1,'single'); 
		        action(1,1,labels(ii+ss),1) = 1;
			input{ss+1} = action;
		    end
		    results = caffe('forward', input);
		    %fprintf('Done with forward pass.\n');
		    recons_image = results{1};
if 0,
		    save([path_to_results sprintf('preds%d_inst%d_session%d_angle%d_t%d.mat',numsteps_prev,cc,tt,ii,numsteps)], 'input', 'images_out', 'recons_image');
end
		    for ss = 1:numsteps,
			recons = recons_image(:,:,(ss-1)*3+1:ss*3,1);
			loss = loss_euclidean(recons, images_out(:,:,(ss-1)*3+1:ss*3,1));
			loss_test_image(ss) = loss_test_image(ss) + loss;
		    end
		    m = m + 1;
            end
	end
    end      
    loss_test_image = loss_test_image ./ m;
    fprintf(sprintf('%s -- testing losses are %f for images in %f seconds.\n', model_specs, mean(loss_test_image), toc));
if 1,
    for ss = 1:numsteps, fprintf(fid_test, '%d %f\n', ss, loss_test_image(ss)); end 
end

end

if 1,
fclose(fid_test);
end
