% infer labels on superpixel MRF for semantic segmentation
imDir = 'examples/pascal_segmentation/VOCdevkit/VOC2010/JPEGImages/';
gtDir = 'examples/pascal_segmentation/VOCdevkit/VOC2010/SegmentationContext59/';
featDir = 'examples/pascal_segmentation/benchmark_RELEASE/dataset/feats/';
clfDir = 'examples/pascal_segmentation/VOCdevkit/local/VOC2010/';
NC = 59;

% init vggnet for feature extraction matcaffe
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

% init superpixel classifier
clfmodel = load([clfDir 'pascal_suerpixel_classification_2_context_iter0025.mat']);
clf(1).w = clfmodel.weights(1).weights{1}; clf(1).b = clfmodel.weights(1).weights{2};
clf(2).w = clfmodel.weights(2).weights{1}; clf(2).b = clfmodel.weights(2).weights{2};

% start inference
valSet = textread('examples/pascal_segmentation/VOCdevkit/VOC2010/ImageSets/Main/val.txt','%s');
predDir = 'examples/pascal_segmentation/VOCdevkit/results/VOC2010/Segmentation/comp6_val_cls/';
if ~isdir(predDir), mkdir(predDir); end;
for ff = 1:length(valSet)
    tic
    name = valSet{ff};
    im = imread([imDir name '.jpg']);
    % extract cnn features on superpixels 
    try
        % load([segDir name '_1024superpixels.mat']);
        % load([featDir name '_1024superpixels.mat']); 
        load([featDir name '.mat']); 
    catch
 	disp('extract features');
        im1 = single(im(:,:,[3 2 1]));
        [im_h, im_w, im_c] = size(im1);
        for c = 1:3, im1(:,:,c) = im1(:,:,c) - mean_pix(c); end
        left = floor((512-im_w)/2); right = 512-im_w-left;
        top = floor((512-im_h)/2); bottom = 512-im_h-top;
        pad = [top, bottom, left, right];
        im1 = imPad(im1, pad, 0);
        [im_h,im_w,im_c] = size(im1);
        im1 = reshape(im1, [im_h,im_w,im_c,1]);
        im1 = permute(im1,[2,1,3,4]);
        output = caffe('forward', {im1});
        acts = caffe('get_all_data');
        segments = SLICSuperpixel(double(im),1024,20);
        save([segDir name '_1024superpixels.mat'], 'segments');
        labels = unique(segments);
        segments1 = imPad(segments, pad, max(labels)+1);
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
                ind_region = find(segments1==labels(k));
                jj_region = jj(ind_region);
                ii_region = ii(ind_region);
                id_region = sub2ind([512/scales(j),512/scales(j)],ii_region,jj_region);
                feats_region = act(:,id_region);
                feats_region = sum(feats_region,2)/length(id_region);
                feats_region = feats_region ./ (norm(feats_region)+eps);
                feats{j}(:,k) = feats_region;		
            end
        end
        save([featDir name '_1024superpixels.mat'], 'feats', 'pad');
    end
    % scoring superpixels
    feats = cell2mat(feats);
    segments = imPad(segments, -pad, 0);
    nSegments = length(unique(segments));
    feats = feats(:,1:nSegments);
    pred1 = clf(1).w'*feats + repmat(clf(1).b,1,nSegments);
    pred1 = pred1.*(pred1>0);
    pred2 = clf(2).w'*pred1 + repmat(clf(2).b,1,nSegments);
    pred2 = 1./(1+exp(-pred2));
%     % build pixel graph
%     unary = -log(pred2);
%     Dc = zeros(size(segments,1),size(segments,2),NC);
%     for cc = 1:NC, tmp = unary(cc,:); Dc(:,:,cc) = tmp(segments); end
%     Sc = 5*(1-eye(NC));%-log(cooc);
%     im = double(im);
%     [Hc1, Vc1] = gradient(imfilter(squeeze(im(:,:,1)),fspecial('gauss',[3 3]),'symmetric'));
%     [Hc2, Vc2] = gradient(imfilter(squeeze(im(:,:,2)),fspecial('gauss',[3 3]),'symmetric'));
%     [Hc3, Vc3] = gradient(imfilter(squeeze(im(:,:,3)),fspecial('gauss',[3 3]),'symmetric'));
%     Hc = Hc1.^2+Hc2.^2+Hc3.^2;
%     Vc = Vc1.^2+Vc2.^2+Vc3.^2;
%     sigma = mean([Hc(:).^.5;Vc(:).^.5]);
%     gch = GraphCut('open', Dc, Sc, single(exp(-.5*Vc/(sigma^2))), single(exp(-.5*Hc/(sigma^2))));
%     [gch, labelmap_pred] = GraphCut('expand',gch);
%     gch = GraphCut('close', gch);
    % build superpixel graph
    unary = -log(pred2);
    pairwise = superpixel2graph(rgb2lab_d65(double(im)), segments, 0);
    % infer MR
    gch = GraphCut('open', unary, 1-eye(NC), sparse(5*pairwise));
    [gch, labels_pred] = GraphCut('expand',gch);
    gch = GraphCut('close', gch);
    labels_pred = labels_pred + 1;
    labelmap_pred = labels_pred(segments);
    % save results
    [gt,cmap] = imread([gtDir name '.png']);
    imwrite(uint8(labelmap_pred), cmap, [predDir name '.png'], 'png');
    fprintf('labeling %04d-th image in %f seconds.\n', ff, toc);
%    figure(1); imshow(uint8(labelmap_pred),cmap); drawnow;
end

addpath('examples/pascal_segmentation/VOCdevkit/VOCcode');
VOCinit;
VOCevalseg(VOCopts, 'comp6');
