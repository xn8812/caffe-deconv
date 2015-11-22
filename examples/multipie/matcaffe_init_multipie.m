function  matcaffe_init_multipie(use_gpu, solver_def_file, device_id)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1 
  % By default use CPU
  use_gpu = 0;
end
if nargin < 2 || isempty(solver_def_file)
  % By default use imagenet_deploy
  solver_def_file = 'examples/multipie/multipie_solver_generator.prototxt';
end
if nargin < 3
  device_id = 0;
end

% caffe('reset');

% if caffe('is_initialized') == 0
%   if exist(model_file, 'file') == 0
%     % NOTE: you'll have to get the pre-trained ILSVRC network
%     error('You need a network model file');
%   end 
%   if ~exist(solver_def_file,'file')
%     % NOTE: you'll have to get network definition
%     error('You need the solver prototxt definition');
%   end 
%   caffe('init', solver_def_file, model_file)
if ~exist(solver_def_file,'file')
    % NOTE: you'll have to get network definition
  error('You need the solver prototxt definition');
end 
caffe('init', solver_def_file);
% end
fprintf('Done with init\n');

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');

if use_gpu
  caffe('set_device', device_id);
  caffe('get_device')
end

caffe('presolve');
