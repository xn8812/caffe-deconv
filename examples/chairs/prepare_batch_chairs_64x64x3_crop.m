% prepare batches for chairs
tic
load('data/chairs/rendered_chairs/all_chair_names.mat');
imfiles = cell(length(folder_names)*length(instance_names),1);
mskfiles = cell(length(folder_names)*length(instance_names),1);
ids = zeros(length(folder_names)*length(instance_names),1);
theta = zeros(1,length(folder_names)*length(instance_names));
phi = zeros(1,length(folder_names)*length(instance_names));
for i=1:length(folder_names)
    for j=1:length(instance_names)
        imfiles{(i-1)*length(instance_names)+j} = ['data/chairs/cropped_chairs/' folder_names{i} '/renders/' instance_names{j}];
        mskfiles{(i-1)*length(instance_names)+j} = ['data/chairs/cropped_chairs/' folder_names{i} '/masks/' instance_names{j}];
        ids((i-1)*length(instance_names)+j) = i;
        pos = strfind(instance_names{j}, '_');
        theta((i-1)*length(instance_names)+j) = str2double(instance_names{j}(pos(2)+2:pos(2)+4));
        phi((i-1)*length(instance_names)+j) = str2double(instance_names{j}(pos(3)+2:pos(3)+4));
    end
end
toc

id_types = unique(ids);
theta_types = unique(theta);
phi_types = unique(phi);


azimuths = [sind(phi);cosd(phi)];
elevations = [sind(theta);cosd(theta)];

% batching
mkdir('examples/chairs/chair_batches_64x64x3_crop');
IMG_HEIGHT = 64;
IMG_WIDTH = 64;
images = zeros(IMG_HEIGHT, IMG_WIDTH, 3,length(id_types), length(theta_types), length(phi_types), 'uint8');
masks = zeros(IMG_HEIGHT, IMG_WIDTH, length(id_types), length(theta_types), length(phi_types), 'uint8');
classes = zeros(length(id_types), length(id_types), length(theta_types), length(phi_types), 'single');
views = zeros(4, length(id_types), length(theta_types), length(phi_types), 'single');
chair_image_mean = zeros(IMG_HEIGHT, IMG_WIDTH,3,'single');
tic
for n = 1 : length(imfiles)
	im = imread(imfiles{n});
	if size(im,3)==1, im = cat(3,im,im,im); end
    im = 255-single(im);
    mask = single(rgb2gray(im)>0);
    im = imresize(im, [IMG_HEIGHT,IMG_WIDTH], 'bilinear');
    mask = imresize(mask, [IMG_HEIGHT,IMG_WIDTH], 'bilinear');
    [k,j,i] = ind2sub([length(phi_types),length(theta_types),length(id_types)], n);
    images(:,:,:,i,j,k) = uint8(im);
    masks(:,:,i,j,k) = uint8(255*mask);
    classes(ids(n),i,j,k) = 1;
    views(:,i,j,k) = [azimuths(:,n);elevations(:,n)];
    chair_image_mean = chair_image_mean*(n-1)/n + im/n;
    if mod(n,62)==0,fprintf([num2str(n) ', ' num2str(toc) '\n']); end
end

% split
samples_test = 3:5:length(phi_types);
samples_train = setdiff(1:length(phi_types),samples_test);
images_train = images(:,:,:,:,:,samples_train);
images_test = images(:,:,:,:,:,samples_test);
masks_train = masks(:,:,:,:,samples_train);
masks_test = masks(:,:,:,:,samples_test);
classes_train = classes(:,:,:,samples_train);
classes_test = classes(:,:,:,samples_test);
views_train = views(:,:,:,samples_train);
views_test = views(:,:,:,samples_test);

% save
mkdir('examples/chairs/chair_batches_64x64x3_crop');
save('examples/chairs/chair_batches_64x64x3_crop/chairs_data_64x64x3_crop.mat', 'images', 'masks', 'classes', 'views');
save('examples/chairs/chair_batches_64x64x3_crop/chair_image_mean_64x64x3_crop.mat', 'chair_image_mean');
save('examples/chairs/chair_batches_64x64x3_crop/chairs_data_train_64x64x3_crop.mat', 'images_train', 'masks_train', 'classes_train', 'views_train');
save('examples/chairs/chair_batches_64x64x3_crop/chairs_data_test_64x64x3_crop.mat', 'images_test', 'masks_test', 'classes_test', 'views_test');

