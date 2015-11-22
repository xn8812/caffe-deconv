% prepare oversampled input
path_to_data  = 'examples/chairs/data/';
load([path_to_data 'chairs_data_64x64x3_crop.mat']);
ids = ids(:); phi = phi(:); theta = theta(:);
train_idx = ids<=500;
images_train = images(:,:,:,train_idx);
[h,w,c,num_train] = size(images_train);
ids_train = ids(train_idx);
classes_train = sparse(double(ids_train(:)),double(1:length(ids_train)),...
    ones(length(ids_train),1,'double'),500,length(ids_train));
classes_train = single(full(classes_train));
phi_train = phi(train_idx);
theta_train = theta(train_idx);
ids_types = unique(ids_train);
phi_types = unique(phi_train);
theta_types = unique(theta_train);


model_specs = 'cnn_chairs';
use_gpu = true;
model_file = sprintf('examples/chairs/%s.prototxt', model_specs);
solver_file = sprintf('examples/chairs/%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.0001, 'stepsize', 200000, 'weight_decay', 0.001, 'solver_type', 0);
make_solver_file(solver_file, model_file, param);
matcaffe_init_chairs(use_gpu, solver_file);
batch_size = 100;


path_to_results = 'examples/chairs/results/';
fid = fopen([path_to_data 'cnn_chairs_errors.txt'],'w');
for n = 1:500
    loss_train_class = 0;
    tic;
    fprintf('Encoder -- processing the %dth iteration.\n',n);
    rnd_idx = randperm(num_train);
    caffe('set_phase_train');
    for m = 1:floor(num_train/batch_size)
        %disp(m)
        batch_idx = rnd_idx((m-1)*batch_size+(1:batch_size));
        images = images_train(:,:,:,batch_idx);
        images = reshape(single(images),[h,w,3,batch_size]);
        %for t=1:batch_size, images(:,:,:,t) = images(:,:,:,t) - chair_image_mean; end
        % images = permute(images(:,:,[3,2,1],:),[2,1,3,4])/255;
        images = permute(images,[2,1,3,4])/255;
        classes = classes_train(:,batch_idx);
        classes = reshape(single(classes),[1,1,size(classes,1),batch_size]);
        preds = caffe('forward', {images});
        preds_class = preds{1};
        [loss_class, delta_class] = loss_crossentropy_paired_softmax_grad(preds_class, classes);
        [~,labels_pred] = max(1./(1+exp(-squeeze(preds_class))),[],1);
        [~,labels_gt] = max(squeeze(classes),[],1);
        loss_class = sum(labels_pred~=labels_gt);
        caffe('backward', {delta_class});
        caffe('update');
        loss_train_class = loss_train_class + loss_class;
    end      
    loss_train_class = loss_train_class / m;
    fprintf(sprintf('Encoder -- total weighted training loss is %f in %f seconds.\n', loss_train_class, toc));
    
    if mod(n,10)==0,
        weights = caffe('get_weights');
        save(sprintf([path_to_data 'chairs_encoder_model_iter%04d.mat'], n), 'weights');
    end
    
end
fclose(fid);
