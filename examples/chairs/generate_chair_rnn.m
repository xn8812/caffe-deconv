
% prepare oversampled input
path_to_data  = 'examples/chairs/data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
test_idx = (ids>0);
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
model_specs = sprintf('rnn_t%d_generator',numsteps);
use_gpu = true;
model_file = sprintf('examples/chairs/%s.prototxt', model_specs);
solver_file = sprintf('examples/chairs/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.000001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
matcaffe_init_chairs(use_gpu, solver_file, 2);

% load pretested layer weights
if 1,
numsteps_prev = 16;
model = caffe('get_weights');
model_pretrained = load(sprintf('examples/chairs/results/rnn_t%d_finetuning_final_model_iter0050.mat',numsteps_prev));
mapping = [];
for i = 1:numsteps, mapping = [mapping, [(1:11)+(i-1)*11; 8:18]]; end
for i = 1:size(mapping,2), model(mapping(1,i)).weights = model_pretrained.weights(mapping(2,i)).weights; end
caffe('set_weights', model);
end

path_to_results = 'examples/chairs/results/';
path_to_feas = 'examples/chairs/results/feas/';
for c1 = 523%[506,504,523,534], % class A
for c2 = 534%[506,504,523,534], % class B
tt = 20; % elevation 
ii = 8; % input view
label = 1; % rotation direction
data1 = load([path_to_feas sprintf('feas_inst%d_ele%d.mat',c1,tt)]);
data2 = load([path_to_feas sprintf('feas_inst%d_ele%d.mat',c2,tt)]);
coeffs = 0.0:0.2:1.0;
tic
for jj = 1:length(coeffs),
	fea_id = (1-coeffs(jj)) * data1.results{1}(:,:,:,ii) + coeffs(jj) * data2.results{1}(:,:,:,ii);
	fea_view = (1-coeffs(jj)) * data1.results{2}(:,:,:,ii) + coeffs(jj) * data2.results{2}(:,:,:,ii);
	action = zeros(1,1,3,1,'single'); 
	action(1,1,label,1) = 1;
	input = cell(2+numsteps,1);
	input{1} = fea_id;
	input{2} = fea_view;
	if label == 1, idx_rot = [ii+1:batch_size,1:ii]; end
	if label == 3, idx_rot = [fliplr(1:ii-1),fliplr(ii:batch_size)]; end
	for ss = 1:numsteps, input{ss+2} = action; end
	results = caffe('forward', input);
	%fprintf('Done with forward pass.\n');
	recons_image = results{1};
	recons_mask = results{2};
	save([path_to_results sprintf('interpolate%d_inst%d_inst%d_ele%d_azi%d_act%d.mat',jj,c1,c2,tt,ii,label)], 'input', 'recons_image', 'recons_mask');
end
fprintf('--interpolate class %d and class %d in %f seconds\n',c1, c2, toc);
end
end
