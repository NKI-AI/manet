@echo off
' Use the following command to run the script in the background
' !start "MATLAB test" /Min /B test_bat.bat
matlab -noFigureWindows -logfile output.txt -r "try; run('C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\MCs_code\find_annotation_4.m'); catch; end; quit"
exit