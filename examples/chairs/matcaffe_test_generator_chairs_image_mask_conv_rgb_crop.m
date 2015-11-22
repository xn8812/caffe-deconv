% test chair generator 64x64
matcaffe_init_chairs(1,'examples/chairs/chairs_solver_generator_image_mask_conv_rgb_crop_1393.prototxt');
batch_size = 100;
prm = struct('hasChn',1);

% prepare oversampled input
path_to_data  = 'examples/chairs/chair_batches_64x64x3_crop/';
load([path_to_data 'chairs_data_test_64x64x3_crop.mat']);
load([path_to_data 'chair_image_mean_64x64x3_crop.mat']);
% input_data is Height x Width x Channel x Num
% do forward pass to get scores
% scores are now Width x Height x Channels x Num
[h,w,c,n2,n3,n4] = size(images_test);
[dim2,n2,n3,n4] = size(classes_test);
[dim3,n2,n3,n4] = size(views_test);

load('models/chairs_generator_image_mask_conv_rgb_crop_model_1393.mat');
caffe('set_weights',weights);
caffe('set_phase_test');

% viewpoint interpolation
% chair id 
for id = 100
    class = zeros(dim2,1,'single'); class(id) = 1;
    classes = repmat(class,[1,batch_size]);
    classes = reshape(classes, [1,1,dim2,batch_size]);
    theta = 20*ones(1,batch_size); phi = 0:360/(batch_size-1):360;
    views = [sind(phi); cosd(phi); sind(theta); cosd(theta)];
    views = reshape(single(views), [1,1,dim3,batch_size]);

    recons = caffe('forward', {classes;views});
    recons_image = recons{1};
    recons_mask = recons{2};
    recons_image_masked = recons_image .* repmat(recons_mask,[1,1,3,1]);
    recons_image_masked = permute(recons_image_masked,[2,1,3,4]);
    for i=1:size(recons_image_masked,4),
        tmp=recons_image_masked(:,:,:,i);
        tmp=tmp-min(tmp(:));tmp=tmp/max(tmp(:));recons_image_masked(:,:,:,i)=tmp; 
    end
    figure(1); montage2(1-recons_image_masked, prm); 
%     set(gcf, 'PaperpositionMode', 'auto');
%     print('-deps', '-r0', 'test.eps');
    pause;
end

% style interpolation
% chair id 
for id_a = 1
    for id_b = 2
        classes = zeros(dim2,batch_size,'single'); 
        classes(id_a,:) = 1:-1/(batch_size-1):0; 
        classes(id_b,:) = 0:1/(batch_size-1):1;
        classes = reshape(classes, [1,1,dim2,batch_size]);
        theta = 20*ones(1,batch_size); phi = 225*ones(1,batch_size);
        views = [sind(phi); cosd(phi); sind(theta); cosd(theta)];
        views = reshape(single(views), [1,1,dim3,batch_size]);

        recons = caffe('forward', {classes;views});
        recons_image = recons{1};
        recons_mask = recons{2};
        recons_image_masked = recons_image .* repmat(recons_mask,[1,1,3,1]);
        recons_image_masked = permute(recons_image_masked,[2,1,3,4]);
        for i=1:size(recons_image_masked,4),
            tmp=recons_image_masked(:,:,:,i);
            tmp=tmp-min(tmp(:));tmp=tmp/max(tmp(:));recons_image_masked(:,:,:,i)=tmp; 
        end
        figure(1); montage2(1-recons_image_masked, prm); 
    %     set(gcf, 'PaperpositionMode', 'auto');
    %     print('-deps', '-r0', 'test.eps');
        pause;
    end
end
