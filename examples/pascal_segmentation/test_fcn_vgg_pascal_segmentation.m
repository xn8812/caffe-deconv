% test semantic segmentation with vggnet fcn
addpath('../caffe-future/examples/pascal_segmentation/VOCdevkit/VOCcode');
model_specs = 'fcn-vgg-unpool-decoder';
use_gpu = true;
model_file = sprintf('examples/pascal_segmentation/%s-test.prototxt', model_specs);
solver_file = sprintf('examples/rgbd/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.00001, 'stepsize', 100000, 'weight_decay', 0.001, 'solver_type', 3);
make_solver_file(solver_file, model_file, param);
mean_pix = [103.939, 116.779, 123.68];
matcaffe_fcn_vgg_init(use_gpu, solver_file);
caffe('set_device', 2);
caffe('set_phase_test');

%imnames = textread('../caffe-future/examples/pascal_segmentation/VOCdevkit/VOC2012/ImageSet/Segmentation/val.txt', '%s');
imnames = textread('examples/pascal_segmentation/data/valsubset.txt', '%s');
mkdir('../caffe-future/examples/pascal_segmentation/VOCdevkit/results/VOC2012/Segmentation/comp5_val_cls');
H = 512; W = 512; NC = 21;

for iter = 38
  tic
  loss_test = 0;
  error_test = 0;
  vggnet = load(sprintf('examples/pascal_segmentation/results/%s_full_model_iter%03d.mat', model_specs, iter));
  caffe('set_weights', vggnet.weights);
  rnd_idx = 1:length(imnames);%randperm(length(imnames));
  for i = 1:length(imnames),
    name = imnames{rnd_idx(i)};
    im = imread(['examples/pascal_segmentation/data/JPEGImages/' name '.jpg']);
%    im = imread(['data/pascal_segmentation/VOC2012/dataset/img/' name '.jpg']);
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
  
%    label = load(['data/pascal_segmentation/VOC2012/dataset/cls/' name '.mat']);
%    label = label.GTcls.Segmentation;
    [label,cmap] = imread(['examples/pascal_segmentation/data/SegmentationClass/' name '.png']);
    [~, label_pred] = max(pred, [], 3);
    label_pred = permute(label_pred,[2,1])-1;
    label_pred = imPad(label_pred, -pad, 0);
%    label_pred = imresize(label_pred,size(label),'nearest'); 
    imwrite(uint8(label_pred),cmap,sprintf('../caffe-future/examples/pascal_segmentation/VOCdevkit/results/VOC2012/Segmentation/comp5_val_cls/%s.png',name),'png');
%    imwrite(uint8(label),cmap,sprintf('examples/pascal_segmentation/results/preds_iter%03d/%s_gt.png',iter,name),'png');
    error_test = error_test + sum(sum(label_pred~=label));
    if mod(i,100)==0, fprintf('process %d images\n', i); end
  end
  error_test  = error_test / length(imnames);
  loss_test = loss_test / length(imnames);
  fprintf('Test error is %f with loss %f in %f seconds.\n', error_test, loss_test, toc);
end
VOCinit;
VOCevalseg(VOCopts,'comp5');
