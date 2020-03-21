function [cropped_new,SegmentationArea] = edge_detection_vs_gaussian5(cropped,cropped_old,patch_size)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
format long g;
format compact;
fontSize = 10;


meanval = mean2 (cropped);
stdval = std2 (cropped);
snr = (meanval / stdval);


hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(cropped), hy, 'replicate');
Ix = imfilter(double(cropped), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);


max_value_grandmag=max(gradmag(:));
mean_value_grandmag=mean(gradmag(:));

g = gradmag - min(gradmag(:));
g = g / max(g(:));


th = graythresh(g); %# Otsu's method.
a = imhmax(g,th/2); %# Conservatively remove local maxima.
th = graythresh(a);
b = a > th; %# Conservative global threshold.


title('b', 'FontSize', fontSize);
SE = strel('square', 1)
c = imclose(b,SE); %# Try to close contours.

d = imfill(c,'holes'); %# Not a bad segmentation by itself.


central_ppixel_value=d(patch_size+1,patch_size+1); 

internal_marker=d;
se3 = strel('disk', 1);
external_marker = imopen(internal_marker,se3);
%     
se4 = strel('disk', 1);
external_marker = imdilate(external_marker,se4);
    
% Then using internal marker, obtain the external marker.
external_marker = ~internal_marker & external_marker;   
     
    
dist_internal_marker = bwdist(~internal_marker);
dist_internal_marker(dist_internal_marker <= 1) = 0;


cropped_new= zeros(size(cropped), 'logical');
CC = bwconncomp(dist_internal_marker);
numPixels = cellfun(@numel,CC.PixelIdxList);
  
%%finde the maximum size CC
[biggest,idx] = max(numPixels);

if CC.NumObjects>0
    S = regionprops('table',CC,'BoundingBox','Extent','PixelList','Area');
    BB=S.BoundingBox;
    Pixel_list=S.PixelList;
    bboxB = [patch_size-2,patch_size-2,5,5];
    Area=S.Area;
    [Max_Area,Index_Area]=max(Area);
    [Ordered_area,index_order] = sort(Area,'descend') 
    Max_index=index_order(1);

    values=CC.PixelIdxList{index_order(1)};
    overlapRatio = bboxOverlapRatio(BB,bboxB);
    [Max_Overlap,Index_overlap]=max(overlapRatio);
    if Max_index==Index_overlap
        cropped_new(values) = 1;
    else
   
        values=CC.PixelIdxList{Index_overlap};
        %values=CC.PixelIdxList{index_order(2)};
        cropped_new(values) = 1;
    
    end
  
%%%%%%%

    S2 = regionprops('table',cropped_new,'Area');
    SegmentationArea=S2.Area;


    cc= bwconncomp(cropped_new)
    stats = regionprops('table',cropped_new,'Centroid','MajorAxisLength','MinorAxisLength','Orientation','Solidity','Eccentricity');
    centroidX=stats.Centroid(1,1);
    centroidY=stats.Centroid(1,2);
    orientation=stats.Orientation(1);
    lenghtMajor=stats.MajorAxisLength(1);
    lenghtMinor=stats.MinorAxisLength(1);
    ratio=lenghtMajor/lenghtMinor;
    soli=stats.Solidity(1);
    ecce=stats.Eccentricity(1);




    %%%%%%%%%%%%%%%%Generate the Gaussian and then segment it%%%%%
    ratio=sqrt(lenghtMajor)/sqrt(lenghtMinor);
    gauss_image=customgauss([patch_size*2+1,patch_size*2+1], ratio*2, 2, orientation, 0, 1, [0 0]);
    % subplot (3,3,8); imshow(gauss_image)
    % title('Gaussian image ', 'FontSize', fontSize);

    hy = fspecial('sobel');
    hx = hy';
    Iy = imfilter(double(gauss_image), hy, 'replicate');
    Ix = imfilter(double(gauss_image), hx, 'replicate');
    gradmag_gauss = sqrt(Ix.^2 + Iy.^2);
    %L = watershed(gradmag);


    %subplot (3,4,4);
%   imshow(gradmag_gauss,[]);
    %title('gradmag', 'FontSize', fontSize);
    %# Normalize.
    g_gauss = gradmag_gauss  - min(gradmag_gauss (:));
    g_gauss = g_gauss / max(g_gauss(:));
    %subplot (3,4,5);
    %imshow(g_gauss,[]);
    %title('Normalized gradmag', 'FontSize', fontSize);

    th_gauss = graythresh(g_gauss); %# Otsu's method.
    a_gauss = imhmax(g_gauss,th_gauss/2); %# Conservatively remove local maxima.
    th_gauss = graythresh(a_gauss);
    b_gauss = a_gauss > th_gauss; %# Conservative global threshold.



    SE = strel('square', 1);
    c_gauss = imclose(b_gauss,SE); %# Try to close contours.
    d_gauss = imfill(c_gauss,'holes'); %# Not a bad segmentation by itself.




    internal_marker_gauss=d_gauss;
    %Finally we are ready to compute the watershed-based segmentation.   
    dist_internal_marker_gauss = bwdist(~internal_marker_gauss);
    dist_internal_marker_gauss(dist_internal_marker_gauss <= 2) = 0;
    %J = imopen(dist_internal_marker_gauss,SE);
    % imshow(dist_internal_marker_gauss,[])
    % figure;

    %if (SegmentationArea<7 | (ecce>0.82) |(SegmentationArea >65 & SegmentationArea <132))
    if (SegmentationArea<7 | (SegmentationArea<15 & ecce>0.82)| (SegmentationArea>40 & ecce>0.84)| lenghtMajor>10 | ecce>0.87)

        messageToShow='Prefer the Gaussian segmentation!';
        cropped_new=imbinarize(dist_internal_marker_gauss);
        segmentation_to_show=cropped_new;
    
    else
        messageToShow='Gaussian segmentation'
        segmentation_to_show=dist_internal_marker_gauss;
    end

else
    
    orientation=0
    ratio=1;
    gauss_image=customgauss([patch_size*2+1,patch_size*2+1], ratio*2, 2, orientation, 0, 1, [0 0]);
    % subplot (3,3,8); imshow(gauss_image)
    % title('Gaussian image ', 'FontSize', fontSize);

    hy = fspecial('sobel');
    hx = hy';
    Iy = imfilter(double(gauss_image), hy, 'replicate');
    Ix = imfilter(double(gauss_image), hx, 'replicate');
    gradmag_gauss = sqrt(Ix.^2 + Iy.^2);
    %L = watershed(gradmag);


    %subplot (3,4,4);
%   imshow(gradmag_gauss,[]);
    %title('gradmag', 'FontSize', fontSize);
    %# Normalize.
    g_gauss = gradmag_gauss  - min(gradmag_gauss (:));
    g_gauss = g_gauss / max(g_gauss(:));
    %subplot (3,4,5);
    %imshow(g_gauss,[]);
    %title('Normalized gradmag', 'FontSize', fontSize);

    th_gauss = graythresh(g_gauss); %# Otsu's method.
    a_gauss = imhmax(g_gauss,th_gauss/2); %# Conservatively remove local maxima.
    th_gauss = graythresh(a_gauss);
    b_gauss = a_gauss > th_gauss; %# Conservative global threshold.



    SE = strel('square', 1);
    c_gauss = imclose(b_gauss,SE); %# Try to close contours.
    d_gauss = imfill(c_gauss,'holes'); %# Not a bad segmentation by itself.




    internal_marker_gauss=d_gauss;
    %Finally we are ready to compute the watershed-based segmentation.   
    dist_internal_marker_gauss = bwdist(~internal_marker_gauss);
    dist_internal_marker_gauss(dist_internal_marker_gauss <= 2) = 0;
    
    cropped_new=imbinarize(dist_internal_marker_gauss);
    segmentation_to_show=cropped_new;
    SegmentationArea=0;
    

end



