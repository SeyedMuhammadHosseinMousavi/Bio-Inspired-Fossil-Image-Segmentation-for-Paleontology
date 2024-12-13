clear;
I=imread('tst2.jpg');
img=I;
I=rgb2gray(I);
% Adjust
I=imadjust(I);
% histeq
J = histeq(rgb2gray(img));
% Edge 
BW2 = edge(I,'canny');
% Otsu
thresh = multithresh(I,5);
seg_I = imquantize(I,thresh);
RGB = label2rgb(seg_I); 	 
% watereshed
bw=imbinarize(I);
D = bwdist(~bw);
D = -D;
L = watershed(D);
L(~bw) = 0;
rgb = label2rgb(L,'jet',[.5 .5 .5]);
% kmeans
he = img;
lab_he = rgb2lab(img);
ab = lab_he(:,:,2:3);
ab = im2single(ab);
nColors = 5;
% repeat the clustering 3 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
km = label2rgb(pixel_labels); 
% title('Image Labeled by Cluster Index');
mask1 = pixel_labels==1;
cluster1 = he .* uint8(mask1);
mask2 = pixel_labels==2;
cluster2 = he .* uint8(mask2);
mask3 = pixel_labels==3;
cluster3 = he .* uint8(mask3);
% imshow(img);
figure;
subplot(2,3,1)
imshow(BW2);title('Canny Edges','FontSize', 13,'color','b');
subplot(2,3,2)
imshow(I);title('Adjust Intensity','FontSize', 13,'color','b');
subplot(2,3,3)
imhist(I,32);title('Histogram Equalization','FontSize', 13,'color','b');
subplot(2,3,4)
imshow(RGB);title('Otsu','FontSize', 13,'color','b');
subplot(2,3,5)
imshow(rgb);title('Watershed','FontSize', 13,'color','b');
subplot(2,3,6)
imshow(km,[]);title('K-Means','FontSize', 13,'color','b');


