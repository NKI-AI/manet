
function [X,Y] = calculate_ellipse(cx, cy, a, b, rotAngle)
    %# This functions returns points to draw an ellipse
    %#
    %#  @param x     X coordinate
    %#  @param y     Y coordinate
    %#  @param a     Semimajor axis
    %#  @param b     Semiminor axis
    %#  @param cx    cetner x position
    %#  @param cy    cetner y position
    %#  @param angle Angle of the ellipse (in degrees)
    %#

    steps = 30;
    angle = linspace(0, 2*pi, steps);

    % Parametric equation of the ellipse
    X = a * cos(angle);
    Y = b * sin(angle);

    % rotate by rotAngle counter clockwise around (0,0)
    xRot = X*cosd(rotAngle) - Y*sind(rotAngle);
    yRot = X*sind(rotAngle) + Y*cosd(rotAngle);
    X = xRot;
    Y = yRot;

    % Coordinate transform
    X = X + cx;
    Y = Y + cy;
    
    
  
end

