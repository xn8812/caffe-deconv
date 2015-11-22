% train a 59-way classifier for pascal superpixels using matcaffe
gtDir = 'examples/pascal_segmentation/VOCdevkit/VOC2010/SegmentationContext59/';
featDir = 'examples/pascal_segmentation/benchmark_RELEASE/dataset/feats/';
trainSet = textread('examples/pascal_segmentation/VOCdevkit/VOC2010/ImageSets/Main/train.txt','%s');
nTrain = numel(trainSet);
NC = 59;
batch_size = 512;

% init matcaffe
model_def_file = 'examples/pascal_segmentation/clf-pascal-solver-2-context.prototxt';
use_gpu = true;
matcaffe_init_pascal(use_gpu, model_def_file);

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
        segments = imPad(segments, -pad, 0);
        nSegments = length(unique(segments));
        gt = imread([gtDir name '.png']);
        labelHist = zeros(NC,nSegments);
        for cc = 1 : NC, labelHist(cc,:) = accumarray(segments(:), gt(:)==cc); end
        labelHist = labelHist ./ repmat(sum(labelHist,1),NC,1);
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
        [~,labels_gt] = max(squeeze(hists),[],1);
        [~,labels_pred] = max(squeeze(preds),[],1);
        error = sum(labels_gt~=labels_pred);
%         penalties = inv_freq(labels_gt).^.5;
%         penalties = repmat(penalties', NC, 1);
%         penalties = reshape(single(penalties), [1,1,NC,batch_size]);
        [loss, delta] = loss_crossentropy_paired_sigmoid_grad(preds, hists);
        caffe('backward', {delta});
        caffe('update');
        error_train = error_train + error;
        loss_train = loss_train + loss;
    end
    error_train = error_train / nTrain;
    loss_train = loss_train / nTrain;
    fprintf(sprintf('Training error is %f with loss %f in %f seconds.\n', error_train, loss_train, toc));
    
    weights = caffe('get_weights');
    save(sprintf(['examples/pascal_segmentation/VOCdevkit/local/VOC2010/pascal_suerpixel_classification_2_context_iter%04d.mat'], iter), 'weights');
    
end
