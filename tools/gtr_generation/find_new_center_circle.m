function [ points] = find_new_center_circle( image,coordinatesX,coordinateY,max_value,X_old,Y_old )
%FIND_NEW_CENTER Summary of this function goes here
%   Detailed explanation goes here
new_center=0;
points=cell(1,3);
points{1,1}=X_old;
points{1,2}=Y_old ; 
points{1,3}=0;
for i=1:size(coordinatesX,2)
    X=coordinatesX(i);
    Y=coordinateY(i);
    pixel_level=image(Y,X);
    A=1+1;
    if pixel_level> max_value
        max_value=pixel_level;
        points{1,1}=X;
        points{1,2}=Y;
        points{1,3}=1;
    end    
end

