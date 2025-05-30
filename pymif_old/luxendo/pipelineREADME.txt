To convert and fuse luxendo images, follow these steps:

1. run _00_h52tif.py with the right folder where the images are
2. open too opposing views of the same channel in Fiji, use the multi-point tool to click on corresponding landmark points. FIRST on the cam0 THEN on the cam1.
3. Analyze->Measure to obtain positions. Save as landmarks.csv in the "raw" folder
4. Run _01_register_and_fisue.py with the right path to the images and landmarks.csv to fuse the opposing views.
NOTE: you might have to change sigmoid1 and sigmoid2...