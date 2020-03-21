function [] = k_means_and_watershed( grayImage )
% This script is implemented to perform Automatic Color based Segmentation 
% of Skin Lesion using unsupervised Kmeans clustering and Watershed based 
% on color components of L*a*b color space.
%
% Question: How to use this script?
% Ans: Please copy this script in the directory containing Original Images
% with GroundTruth Images then open in the MATLAB(R2016a) and press F5,after 
% processing, it will display the Original Image, its GroundTruth Image, 
% Generated Binary Mask,Segmented RGB Lesion and Calcualted Jaccard Index 
% score for each image in the directory.
% This script is not saving Generated Binary Mask, nor Segmented RGB Lesion 
% in order to save them the script needs to modify.
%            
% Project Title: Automatic Color based Segmentation of Skin Lesion
% Authors: Amjad Khan, Ama Katseena Yawson and Natalia Herrera Murillo
% Supervisor: Professor Alessandro Bria
% Course: Advanced Image Analysis
% Degree: Master in Medical Imaging and Applications
% Date: 03-05-2018
%**************************************************************************
% Start processing the image one by one.
  
    imshow(grayImage);
    grayImage=im2double(grayImage);
    % Transform rgb image to L*a*b color space. 
    %lab_img = rgb2lab(rgb_img);
    
    % For color based K-means, extract *a*b color components from L*a*b 
    % color space.
    %ab = lab_img(:,:,2:3);
    
    % Prepare *a*b color components for K-means clustering algorithm.
    nrows = size(grayImage,1);
    ncols = size(grayImage,2);
    grayImage_forKmeans = reshape(grayImage,nrows*ncols,1);
    
    
    % Decide number of colors/clusters for clustering all the pixels into
    % them.
    nColors = 2;
    
    % Apply Matlab (built-in) K-means function based on squared euclidean 
    % distance.
    [cluster_idx, cluster_center] = kmeans(grayImage_forKmeans,nColors,'distance', ...
        'sqEuclidean');
    
    % Arrange the cluster index to get each pixel label.
    %pixel_labels = reshape(cluster_idx,nrows,ncols);
    pixel_labels=reshape(cluster_idx,nrows,ncols);
    
    % Define cell variables to save obtained clusters.
    clusters = cell(1,2);
%     
%     % Arrange pixel label for all R,G and B color channels.
%     %rgb_label = repmat(pixel_labels,[1 1 3]);
%     
    % Separate each cluster based on color in RGB image and save into new
    % image variables clusters{1} and clusters{2}.
    for j = 1:nColors
        color_img = grayImage;
        color_img(pixel_labels ~= j) = 0;
        clusters{j} = color_img;
    end
%     
%     % Calculate mean intensities in each cluster to distinguish between
%     % skin (high intensity) and lesion (low intensity) regions.
    mean_cluster1 = mean(mean(clusters{1}));
    mean_cluster2 = mean(mean(clusters{2}));
    min_clusters = min(mean_cluster1, mean_cluster2);
    max_clusters = max(mean_cluster1, mean_cluster2);
%     
%     % Decide lesion region (based on minimum intensity compare to skin
%     % region).
    if(min_clusters == mean_cluster1)
        lesion_cluster = clusters{1};
    else lesion_cluster = clusters{2};    
    end
%     
%     % Convert the decided lesion region into gray image for morphological
%     % operations to smoothen and preserve edges especially in case of hairy 
%     % skin lesion.
    gry_lesion_cluster = lesion_cluster;
    se1 = strel('disk', 1);
    se2 = strel('disk', 2);
    %lesion = imopen(gry_lesion_cluster,se1);
    %lesion = imclose(lesion,se2);
    lesion=gry_lesion_cluster;
% 
%     % Prepare markers for Watershed algorithm
%     
%     % Build internal marker from the smoothened binary version of lesion 
%     % region. 
    internal_marker = logical(lesion);
       
    % Remove all unwanted pixels or objects around the lesion region that
    % are smaller in size as compared to lesion.
    objects = bwconncomp(internal_marker, 4);
    area = regionprops(objects , 'Area');
    large_object = struct2cell(area);
    large_object = cell2mat(large_object);
    large_object = max(large_object);
    labels = labelmatrix(objects);
    internal_marker = ismember(labels, find([area.Area] >= large_object));
    internal_marker = imfill(internal_marker,'holes');
       
    % Prepare external marker from internal marker by using opening and 
    % dilation morphological operations.
     se3 = strel('disk', 1);
    external_marker = imopen(internal_marker,se3);
    
    se4 = strel('disk', 2);
    external_marker = imdilate(external_marker,se4);
    
    % Then using internal marker, obtain the external marker.
    external_marker = ~internal_marker & external_marker;

    % Transform internal marker using distance transform and thresholding.    
    dist_interal_marker = bwdist(~internal_marker);
    dist_interal_marker(dist_interal_marker <= 95) = 0;
    
    % Also calculate the distance transform of external marker.
    dist_external_marker = bwdist(~external_marker);

    % Combine markers for Watershed algorithm.    
    markers = dist_interal_marker + dist_external_marker;
    
%     % Convert Original RGB image to grayscale image.
%     %gray_img = rgb2gray(rgb_img);
     gray_img =grayImage;
% 
%     % Obtain local minima from the grayscale image using markers location.
     gray_img = imimposemin(gray_img, markers);
%     
%     % Start displaying the results with inputs.
%     % Display groundtruth image.
     
%     imshow(gt_img);title('Ground Truth')
% 
%     % Calcuate the Watershed (built-in MATLAB function) using imposed 
%     % minima image (with inversion improved performance is obtained).
    WSL = watershed(~gray_img);
    imshow(grayImage);
    figure;
     
    subplot 211
    imshow(grayImage);title('Original image')
%     
%     % Obtain lesion binary mask from the output of the Watershed.
     bw_mask = logical(WSL==2);
%     
%     % Calculate Jaccard index using ground truth with generated binary
%     % mask.
%      gt_mask = logical(gt_img);
%     
%     % Find the intersection of the two images (ground truth and binary
%     % mask).
%      inter_img = gt_mask & bw_mask;
% 
%     % Find the union of the two images (ground truth and binary mask).
%     union_img = gt_mask | bw_mask;
%     
%     % Save calculated Jaccard index score for each image. 
%     jaccardIdx(i,:) = sum(inter_img(:))/sum(union_img(:));
%     
%     % Display generated binary mask with calculated Jaccard index score.
     subplot 212
     imshow(bw_mask)
     s1 = 'Generated Binary Mask (';
%     s2 = num2str(jaccardIdx(i,:));
     title(strcat(s1,'',')'))
       
end

