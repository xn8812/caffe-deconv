% run image classification with vggnet
addpath(genpath('matlab'));
addpath(genpath('examples/pascal_segmentation/SLIC'));

model_def_file = 'examples/pascal_segmentation/fcn-vgg-feature-solver.prototxt';
use_gpu = true;
mean_pix = [103.939, 116.779, 123.68];
matcaffe_init(use_gpu, model_def_file, []);
weights0 = caffe('get_weights');

load('examples/pascal_segmentation/fcn-8s-pascal-weights.mat');
for i=1:15, weights0(i).weights = weights(i).weights; end
caffe('set_weights', weights0);
layers = [3, 6, 10, 14, 18, 21];
scales = [1, 2, 4, 8, 16, 32];

trainset = textread('examples/pascal_segmentation/VOCdevkit/VOC2010/ImageSets/Main/train.txt','%s');
valset = textread('examples/pascal_segmentation/VOCdevkit/VOC2010/ImageSets/Main/val.txt','%s');
imnames = [trainset; valset];
tic
for i = 1:length(imnames),
    name = imnames{i};
    if exist(['examples/pascal_segmentation/benchmark_RELEASE/dataset/feats/' name '.mat'], 'file'),
        fprintf('skip %d images\n', i);
       continue;
    end
    im0 = imread(['examples/pascal_segmentation/VOCdevkit/VOC2010/JPEGImages/' name '.jpg']);
    im = single(im0(:,:,[3 2 1]));
    [im_h, im_w, im_c] = size(im);
    for c = 1:3, im(:,:,c) = im(:,:,c) - mean_pix(c); end
    left = floor((512-im_w)/2); right = 512-im_w-left;
    top = floor((512-im_h)/2); bottom = 512-im_h-top;
    pad = [top, bottom, left, right];
    im = imPad(im, pad, 0);
    [im_h,im_w,im_c] = size(im);
    im = reshape(im, [im_h,im_w,im_c,1]);
    im = permute(im,[2,1,3,4]);
    output = caffe('forward', {im});
    acts = caffe('get_all_data');
    segments = SLICSuperpixel(double(im0),1024,20);
    labels = unique(segments);
    segments = imPad(segments, pad, max(labels)+1);
    [xx,yy] = meshgrid(1:512,1:512);
    xx = xx(:); yy = yy(:);
    ind = sub2ind([512,512], yy, xx);
    
    feats = cell(length(layers),1);
    for j = 1:length(layers),
        jj = ceil(xx/scales(j)); 
        ii = ceil(yy/scales(j));
        act = acts(layers(j)).data;
        [n1,n2,n3] = size(act);
        act = reshape(permute(act,[3,2,1]),[n3,n2*n1]);
        feats{j} = zeros(n3,length(labels),'single');
        for k = 1:length(labels)
            ind_region = find(segments==labels(k));
            jj_region = jj(ind_region);
            ii_region = ii(ind_region);
            id_region = sub2ind([512/scales(j),512/scales(j)],ii_region,jj_region);
            feats_region = act(:,id_region);
            feats_region = sum(feats_region,2)/length(id_region);
            feats_region = feats_region ./ (norm(feats_region)+eps);
            feats{j}(:,k) = feats_region;		
        end
    end
    save(['examples/pascal_segmentation/benchmark_RELEASE/dataset/feats/' name '.mat'], 'segments', 'feats', 'pad');
    if mod(i,10)==0, fprintf('processing %d images in %f seconds.\n', i, toc); end
end
