if 0,
% prepare oversampled input
path_to_data  = 'examples/chairs/data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
test_idx = ids > 500;
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
model_specs = 'VGG_ILSVRC_16_layers_features';
use_gpu = true;
model_file = sprintf('examples/rgbd/%s.prototxt', model_specs);
solver_file = sprintf('examples/rgbd/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.000001, 'stepsize', 100000, 'weight_decay', 0.001, 'solver_type', 0);
make_solver_file(solver_file, model_file, param);
matcaffe_init_vgg(use_gpu, solver_file);

mean_pix = [103.939, 116.779, 123.68];
load('examples/rgbd/VGG_ILSVRC_16_layers_model.mat');
model_init = caffe('get_weights');
model_init = model(1:end-1);
caffe('set_weights',model_init);
batch_size = 1;

feats_test = cell(1,length(ids_types)*length(theta_types));
m = 1;
for cc = ids_types(:)', % chair instance
	tic
	for tt = theta_types(:)', % elevation
	    batch_idx = find(ids_test==cc & theta_test==tt);
	    images_batch = images_test(:,:,:,batch_idx);
	    images_batch = single(permute(images_batch,[2,1,3,4]));    
	    phi_batch = phi_test(batch_idx);
	    [phi_batch, order] = sort(phi_batch, 'ascend');
	    images_batch = images_batch(:,:,:,order);
            feats_test{m} = zeros(4096,length(batch_idx),'single');
	    for jj = 1:length(batch_idx),
		image = imresize(255-images_batch(:,:,:,jj), [224,224], 'bilinear');
		image = image(:,:,[3,2,1]);
		for c = 1:3, image(:,:,c) = image(:,:,c) - mean_pix(c); end
	        results = caffe('forward', {image});
		feats_test{m}(:,jj) = squeeze(results{1});
            end
	    m = m + 1;
	end
	fprintf('--class %d in %f seconds\n',cc, toc);
end      
feats_test = cell2mat(feats_test);
end
feats_test = feats_test ./ repmat(sqrt(sum(feats_test.^2,1)),4096,1);

% gallery and probe split
acc_test = zeros(31,15);
rnd_idx = randperm(length(phi_types(:)));
rnd_idx = rnd_idx(1:10);
for i = 1:31%rnd_idx(:)',

 	select_phi = phi_types(i);
	ind_gallery = (phi_test == select_phi);
	ind_probe = (~ind_gallery);		

	% gallery/probe split
	feats_gallery = feats_test(:,ind_gallery);
	feats_probe = feats_test(:,ind_probe);
	ids_gallery = ids_test(ind_gallery);
	ids_probe = ids_test(ind_probe);
	phi_gallery = phi_test(ind_gallery);
	phi_probe = phi_test(ind_probe);

	% fix phi degree
	delta = 360/31;
	phi_gallery = delta*round(phi_gallery/delta);
	phi_probe = delta*round(phi_probe/delta);
 	select_phi = delta*round(select_phi/delta);
        diff = abs(phi_probe(:) - select_phi);
	diff = (diff>180).*(360-diff) + (diff<=180).*diff;
	diff = round(diff);

	% similarity
	dist = EuDist2(feats_gallery', feats_probe', 1); 
	[~,matches] = min(dist,[],1);
	ids_pred = ids_gallery(matches);

	% breakdown
	diff_types = unique(diff(:));
	acc = [];
	for d = diff_types(:)',
	    ids_pred_sub = ids_pred(diff==d);
	    ids_probe_sub = ids_probe(diff==d);
	    acc = [acc; sum(ids_pred_sub==ids_probe_sub)/length(ids_probe_sub)];
	end
	acc_test(i,:) = acc(:)';
end
save('examples/chairs/results/acc_test_vgg.mat', 'acc_test');
pause;
