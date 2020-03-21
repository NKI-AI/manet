function [points] = find_new_center_bbox( image,max_value,X_old,Y_old,bbox_size )
%FIND_NEW_CENTER Summary of this function goes here
%   Detailed explanation goes here
new_center=0;
points=cell(1,3);
points{1,1}=X_old;
points{1,2}=Y_old ; 
points{1,3}=0;
for i=X_old-bbox_size:X_old+bbox_size
    for j=Y_old-bbox_size:Y_old+bbox_size
     pixel_level=image(j,i);
     if pixel_level> max_value
        max_value=pixel_level;
        points{1,1}=i;
        points{1,2}=j;
        points{1,3}=1;
    end   
     
    end
end


end
