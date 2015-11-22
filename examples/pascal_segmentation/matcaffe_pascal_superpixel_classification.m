% train a 21-way classifier for pascal superpixels using matcaffe
gtDir = 'examples/pascal_segmentation/VOCdevkit/VOC2011/SegmentationClass/';
featDir = 'examples/pascal_segmentation/VOCdevkit/local/VOC2011/feats/';
segDir = 'examples/pascal_segmentation/VOCdevkit/local/VOC2011/superpixels/';
trainSet = textread('examples/pascal_segmentation/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt','%s');
valSet = textread('examples/pascal_segmentation/VOCdevkit/VOC2011/ImageSets/Segmentation/val.txt','%s');
load('examples/pascal_segmentation/VOCdevkit/local/VOC2011/inv_freq.mat');
nTrain = numel(trainSet);
nVal = numel(valSet);
NC = 21;
batch_size = 256;

% init matcaffe
solver_file = 'examples/pascal_segmentation/clf-pascal-solver-2.prototxt';
use_gpu = true;
matcaffe_init(use_gpu, solver_file, []);

%start training
for iter = 1:100
    error_train = 0;
    loss_train = 0;
    tic;
    fprintf('Processing the %dth iteration.\n',iter);
    rndIdx = randperm(nTrain);
    caffe('set_phase_train');
    for ff = 1:nTrain
        name = trainSet{rndIdx(ff)};
        load([featDir name '.mat']);
	load([segDir name '.mat']);
        nSegments = length(unique(segments));
        gt = imread([gtDir name '.png']);
        gt(gt==255) = 21; gt = gt + 1;
        labelHist = zeros(NC+1,nSegments);
        for cc = 1 : NC+1, labelHist(cc,:) = accumarray(segments(:), gt(:)==cc); end
        labelHist = labelHist ./ repmat(sum(labelHist,1),NC+1,1);
        sample = randperm(nSegments);
        sample = sample(1:batch_size);
        labelHist = labelHist(1:NC,:);
        hists = labelHist(:,sample)>0;
        hists = single(reshape(hists,[1,1,NC,batch_size]));             
        feats = cell2mat(feats);
        feats = feats(:,1:nSegments);
        feats = feats(:,sample);
        feats = reshape(feats,[1,1,size(feats,1),batch_size]);
        preds = caffe('forward',{single(feats)});
        preds = preds{1};
	acts = act_softmax(preds);
        [~,labels_gt] = max(squeeze(hists),[],1);
        [~,labels_pred] = max(squeeze(acts),[],1);
        error = sum(labels_gt~=labels_pred);
        penalties = inv_freq(labels_gt).^.5;
        penalties = repmat(penalties', NC, 1);
        penalties = reshape(single(penalties), [1,1,NC,batch_size]);
        [loss, delta] = loss_crossentropy_paired_sigmoid_grad(preds, hists, penalties);
        caffe('backward', {delta});
        caffe('update');
        error_train = error_train + error;
        loss_train = loss_train + loss;
    end
    error_train = error_train / nTrain;
    loss_train = loss_train / nTrain;
    fprintf(sprintf('Training error is %f with loss %f in %f seconds.\n', error_train, loss_train, toc));
    
    weights = caffe('get_weights');
    save(sprintf(['examples/pascal_segmentation/VOCdevkit/local/VOC2011/pascal_suerpixel_classification_2_iter%04d.mat'], iter), 'weights');
    
end
