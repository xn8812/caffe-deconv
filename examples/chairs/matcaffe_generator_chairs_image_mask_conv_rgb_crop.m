function [error_train, error_test] = matcaffe_generator_chairs_image_mask_conv_rgb_crop(use_gpu)

% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init_chairs(use_gpu,'examples/chairs/chairs_solver_generator_image_mask_conv_rgb_crop.prototxt');
else
  matcaffe_init_chairs();
end

batch_size = 100;

% prepare oversampled input
path_to_data  = 'examples/chairs/chair_batches_64x64x3_crop/';
load([path_to_data 'chairs_data_train_64x64x3_crop.mat']);
load([path_to_data 'chairs_data_test_64x64x3_crop.mat']);
load([path_to_data 'chair_image_mean_64x64x3_crop.mat']);
% input_data is Height x Width x Channel x Num
% do forward pass to get scores
% scores are now Width x Height x Channels x Num
images_train = images_train(:,:,:,1:809,:,:);
masks_train = masks_train(:,:,1:809,:,:);
classes_train = classes_train(1:809,1:809,:,:);
views_train = views_train(:,1:809,:,:);
images_test = images_test(:,:,:,1:809,:,:);
masks_test = masks_test(:,:,1:809,:,:);
classes_test = classes_test(1:809,1:809,:,:);
views_test = views_test(:,1:809,:,:);

[h,w,c,n2,n3,n4] = size(images_train);
num_train = n2*n3*n4;
images_train = reshape(images_train, [h,w,c,n2*n3*n4]);
% images_train = images_train - repmat(chair_image_mean, 1, n2*n3*n4);
masks_train = reshape(masks_train, [h,w,n2*n3*n4]);
% images_train = images_train .* masks_train;
[dim2,n2,n3,n4] = size(classes_train);
classes_train = reshape(classes_train, [dim2,n2*n3*n4]);
[dim3,n2,n3,n4] = size(views_train);
views_train = reshape(views_train, [dim3,n2*n3*n4]);

[h,w,c,n2,n3,n4] = size(images_test);
num_test = n2*n3*n4;
images_test = reshape(images_test, [h,w,c,n2*n3*n4]);
% images_test = images_test - repmat(chair_image_mean(:), 1, n2*n3*n4);
masks_test = reshape(masks_test, [h,w,n2*n3*n4]);
% images_test = images_test .* masks_test;
[dim2,n2,n3,n4] = size(classes_test);
classes_test = reshape(classes_test, [dim2,n2*n3*n4]);
[dim3,n2,n3,n4] = size(views_test);
views_test = reshape(views_test, [dim3,n2*n3*n4]);


for n = 1:500
    error_train = 0;
    tic;
    fprintf('Processing the %dth iteration.\n',n);
    rnd_idx = randperm(num_train);
    caffe('set_phase_train');
    for m = 1:floor(num_train/batch_size)
        %disp(m)
        batch_idx = rnd_idx((m-1)*batch_size+(1:batch_size));
        classes = classes_train(:,batch_idx);
        classes = reshape(classes, [1,1,dim2,batch_size]);
        views = views_train(:,batch_idx);
        views = reshape(views, [1,1,dim3,batch_size]);
        images = images_train(:,:,:,batch_idx);
        images = reshape(images,[h,w,3,batch_size]);
        images = single(permute(images,[2,1,3,4]))/255;
        masks = masks_train(:,:,batch_idx);
        masks = reshape(masks,[h,w,1,batch_size]);
        masks = single(permute(masks,[2,1,3,4]))/255;
        recons = caffe('forward', {classes;views});
        %fprintf('Done with forward pass.\n');
%         act = caffe('get_all_data');
        recons_image = recons{1};
        recons_mask = recons{2};
        loss_image = loss_euclidean(recons_image, images);
        delta_image = loss_euclidean_grad(recons_image, images);
        loss_mask = loss_euclidean(recons_mask, masks);
        delta_mask = loss_euclidean_grad(recons_mask, masks);
        caffe('backward', {delta_image; delta_mask});
%         diff = caffe('get_all_diff');
        caffe('update');
        %fprintf('Done with update\n');
        %save diff_cifar.mat d;
        loss = 10*loss_image + loss_mask;
        error_train = error_train + loss;
    end      
    if mod(n,100)==0
        images = permute(images, [2,1,3,4]); 
        recons_image_masked = recons_image .* repmat(recons_mask,[1,1,3]);
        recons_image_masked = permute(recons_image_masked, [2,1,3,4]); 
        save(sprintf([path_to_data 'chairs_generator_image_mask_conv_rgb_crop_train_samples_iter%04d.mat'], n), 'images', 'recons_image_masked');
%         figure(1); subplot(1,2,1); montage2(images); subplot(1,2,2); montage2(recons_image_masked); 
%     %     title('training'); drawnow; 
%         set(gcf, 'PaperpositionMode', 'auto');
%         print('-deps', '-r0', 'train.eps');
%         pause(1);
    end
	error_train = error_train / m;
	fprintf(sprintf('Training error is %f in %f seconds.\n', error_train, toc));
    
    if mod(n,100)==0,
        weights = caffe('get_weights');
        save(sprintf([path_to_data 'chairs_generator_image_mask_conv_rgb_crop_model_iter%04d.mat'], n), 'weights');
    end
    

    caffe('set_phase_test');
	error_test = 0;
    rnd_idx = randperm(num_test);
    for m = 1:floor(num_test/batch_size)
        batch_idx = rnd_idx((m-1)*batch_size+(1:batch_size));
        classes = classes_test(:,batch_idx);
        classes = reshape(classes, [1,1,dim2,batch_size]);
        views = views_test(:,batch_idx);
        views = reshape(views, [1,1,dim3,batch_size]);
        images = images_test(:,:,:,batch_idx);
        images = reshape(images,[h,w,3,batch_size]);
        images = single(permute(images,[2,1,3,4]))/255;
        masks = masks_test(:,:,batch_idx);
        masks = reshape(masks,[h,w,1,batch_size]);
        masks = single(permute(masks,[2,1,3,4]))/255;
        
        recons = caffe('forward', {classes;views});
        recons_image = recons{1};
        recons_mask = recons{2};
        loss_image = loss_euclidean(recons_image, images);
        loss_mask = loss_euclidean(recons_mask, masks);
        loss = 10*loss_image + loss_mask;
        error_test = error_test + loss;
    end
    if mod(n,100)==0,
        images = permute(images, [2,1,3,4]); 
        recons_image_masked = recons_image .* repmat(recons_mask,[1,1,3]);
        recons_image_masked = permute(recons_image_masked, [2,1,3,4]); 
        save(sprintf([path_to_data 'chairs_generator_image_mask_conv_rgb_crop_test_samples_iter%04d.mat'], n), 'images', 'recons_image_masked');
%         figure(2); subplot(1,2,1); montage2(images); subplot(1,2,2); montage2(recons_image_masked); 
%     %     title('test'); drawnow; 
%         set(gcf, 'PaperpositionMode', 'auto');
%         print('-deps', '-r0', 'test.eps');
%         pause(1);
    end
	error_test = error_test / m;
	fprintf(sprintf('Test error is %f.\n', error_test));
end
