% train semantic segmentation with vggnet fcn
model_specs = 'fcn-vgg-unpool-decoder';
use_gpu = true;
model_file = sprintf('examples/pascal_segmentation/%s.prototxt', model_specs);
solver_file = sprintf('examples/rgbd/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.00001, 'lr_policy', 'fixed', 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
mean_pix = [103.939, 116.779, 123.68];
matcaffe_fcn_vgg_init(use_gpu, solver_file);
weights0 = caffe('get_weights');
vggnet = load('examples/pascal_segmentation/fcn-8s-pascal-weights.mat');
for i=1:15, weights0(i).weights = vggnet.weights(i).weights; end
caffe('get_device')
caffe('set_weights', weights0);
caffe('set_phase_train');
load(sprintf('examples/pascal_segmentation/results/%s_large_model_iter%03d.mat', model_specs, 12), 'weights');
caffe('set_weights', weights');

imnames = textread('examples/pascal_segmentation/data/train.txt', '%s');
% imnames = textread('../caffe-future/examples/pascal_segmentation/benchmark_RELEASE/dataset/train.txt', '%s');
% imnames = textread('../caffe-future/examples/pascal_segmentation/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt', '%s');
length(imnames)
pause;

stat = load('examples/pascal_segmentation/inv_freq.mat');
inv_freq = [stat.inv_freq;0];
H = 512; W = 512; NC = 21;

fid = fopen(sprintf('examples/pascal_segmentation/results/%s-train-full-errors.txt', model_specs),'w');
for iter = 1 : 100
  tic
  loss_train = 0;
  error_train = 0;
  rnd_idx = 1:length(imnames);%randperm(length(imnames));
  for i = 1:length(imnames),
    name = imnames{rnd_idx(i)};
%    im = imread(['../caffe-future/examples/pascal_segmentation/VOCdevkit/VOC2011/JPEGImages/' name '.jpg']);
%    im = imread(['../caffe-future/examples/pascal_segmentation/benchmark_RELEASE/dataset/img/' name '.jpg']);
    im = imread(['examples/pascal_segmentation/data/JPEGImages/' name '.jpg']);
%    im = imresize(im,0.5,'bilinear');
    im = single(im(:,:,[3 2 1]));
    [im_h, im_w, im_c] = size(im);
    for c = 1:3, im(:,:,c) = im(:,:,c) - mean_pix(c); end
    left = floor((W-im_w)/2); right = W-im_w-left;
    top = floor((H-im_h)/2); bottom = H-im_h-top;
    pad = [top, bottom, left, right];
    im = imPad(im, pad, 0);
    [im_h,im_w,im_c] = size(im);
    im = reshape(im, [im_h,im_w,im_c,1]);
    im = permute(single(im),[2,1,3,4]);
    output = caffe('forward', {im});
    pred = output{1};
  
    label = imread(['examples/pascal_segmentation/data/SegmentationClass/' name '.png']);
%    label = load(['../caffe-future/examples/pascal_segmentation/benchmark_RELEASE/dataset/cls/' name '.mat']);
%    label = label.GTcls.Segmentation;
%    [label,cmap] = imread(['../caffe-future/examples/pascal_segmentation/VOCdevkit/VOC2011/SegmentationClass/' name '.png']);
    label(label==255) = 21;
%    label = imresize(label,0.5,'nearest');
    label = imPad(label, pad, 0);
    label = label + 1;
    label = permute(label, [2,1]);
    ids = unique(label); ids(ids==22)=[];
    cls = zeros(H,W,NC,1,'single');
    for c = 1:length(ids), cls(:,:,ids(c),:) = single(label==ids(c)); end

    penalties = inv_freq(label).^.5;
    penalties = repmat(penalties, 1,1,NC);
    penalties = reshape(single(penalties), [H,W,NC]);

    [loss, delta] = loss_crossentropy_paired_sigmoid_grad(pred, cls, penalties);
    delta = reshape(single(delta),[H,W,NC,1]);
    caffe('backward', {delta});
    caffe('update');
    loss_train = loss_train + loss;
    [~, label_pred] = max(pred, [], 3);
    error_train = error_train + sum(sum(label_pred~=label));
  end
  error_train  = error_train / length(imnames);
  loss_train = loss_train / length(imnames);
  fprintf('Iter %d: training error is %f with loss %f in %f seconds.\n', iter, error_train, loss_train, toc);
  fprintf(fid, '%d %f\n', iter, error_train);
  weights = caffe('get_weights');
  save(sprintf('examples/pascal_segmentation/results/%s_full_model_iter%03d.mat', model_specs, iter), 'weights');
end
fclose(fid);
