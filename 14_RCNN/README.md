# RCNN and Depth Assignment

You must have 100 background, 100x2 (including flip), and you randomly place the foreground on the background 20 times, you have in total 100x200x20 images.

In total you MUST have:

1.  400k fg_bg images
2.  400k depth images
3.  400k mask images
4.  generated from:
    1.  100 backgrounds
    2.  100 foregrounds, plus their flips
    3.  20 random placement on each background.
5.  Now add a readme file on GitHub for Project 15A:
    1.  Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file.
    2.  Add your dataset statistics:
        1.  Kinds of images (fg, bg, fg_bg, masks, depth)
        2.  Total images of each kind
        3.  The total size of the dataset
        4.  Mean/STD values for your fg_bg, masks and depth images
    3.  Show your dataset the way I have shown above in this readme
    4.  Explain how you created your dataset
        1.  how were fg created with transparency
        2.  how were masks created for fgs
        3.  how did you overlay the fg over bg and created 20 variants
        4.  how did you create your depth images?
6.  Add the notebook file to your repo, one which you used to create this dataset
7.  Add the notebook file to your repo, one which you used to calculate statistics for this dataset

Things to remember while creating this dataset:

1.  stick to square images to make your life easy.
2.  We would use these images in a network which would take an fg_bg image AND bg image, and predict your MASK and Depth image. So the input to the network is, say, 224x224xM and 224x224xN, and the output is 224x224xO and 224x224xP.
3.  pick the resolution of your choice between 150 and 250 for ALL the images

# Solution

## Dataset Google Drive link

https://drive.google.com/open?id=10MBvlf6pMB78o-bWNe7tVlqNaIP3DtKQ

.`zip`, no compression algorithm was used, `ZIP_STORE` option was used

### Total Size
![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/images/dataset-size.png?raw=true)

## Dataset Creation

Github Link: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/01_DenseDepth_DatasetCreation.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/01_DenseDepth_DatasetCreation.ipynb)

[https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/01_02_DenseDepth_DatasetCreation.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/01_02_DenseDepth_DatasetCreation.ipynb)

Colab Link: https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/01_DenseDepth_DatasetCreation.ipynb

[https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/01_02_DenseDepth_DatasetCreation.ipynb](https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/01_02_DenseDepth_DatasetCreation.ipynb)

## Depth Map creation

Colab link: [https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/02_DepthModel_DepthMap.ipynb](https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/02_DepthModel_DepthMap.ipynb)

## Mean and Standard Deviation

Github Link: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/03_DepthModel_MeanStd.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/03_DepthModel_MeanStd.ipynb)

Colab Link: https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/03_DepthModel_MeanStd.ipynb

### Dataset Stats:

1. BG Images

- Mean:`['0.573435604572296', '0.520844697952271', '0.457784473896027']`
- Std: `['0.207058250904083', '0.208138316869736', '0.215291306376457']`

2. FG_BG Images

- Mean: `['0.568499565124512', '0.512103974819183', '0.452332496643066']`
- Std:  `['0.211068645119667', '0.211040720343590', '0.216081097722054']`

3. FG_BG_MASK Images

- Mean: `['0.062296919524670', '0.062296919524670', '0.062296919524670']`
- Std: `['0.227044790983200', '0.227044790983200', '0.227044790983200']`

4. DEPTH_FG_BG

- Mean: `['0.302973538637161', '0.302973538637161', '0.302973538637161']`
- Std: `['0.101284727454185', '0.101284727454185', '0.101284727454185']`

## Dataset Visualization

Github Link: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/04_DepthModel_DataViz.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/04_DepthModel_DataViz.ipynb)

Colab Link: https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/04_DepthModel_DataViz.ipynb

Note: To view them larger, `right click -> Open image in new tab`

### BG Images

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/images/bg.png?raw=true)

### FG Images

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/images/fg.png?raw=true)

### FG_BG Images

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/images/fg_bg.png?raw=true)

### FG_BG_MASK Images

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/images/fg_bg_mask.png?raw=true)

### Depth_FG_BG Images

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/14_RCNN/images/depth_fg_bg.png?raw=true)

# How the dataset was created

Since we need to apply the foreground images on background images and also creating a mask of the fg images, i used transparent background png images, a image crawler was run on Bing to gather people, animals, dogs, cats, bears, goats, deer, cow and human images for the fg, mall interior, interior and indoor images were searched and crawled.

Now we converted the fg png images to mask by filling the transparent part with white and rest image with black using this code,

```python
img  =  cv2.imread(fg_images[0],  cv2.IMREAD_UNCHANGED)
ret,  mask  =  cv2.threshold(im[:,  :,  3],  0,  255,  cv2.THRESH_BINARY)
```

For the BG Images, they were resized and cropped to `200x200`  using this,

```python
def  crop_center(pil_img):
	img_width,  img_height  =  pil_img.size
	crop_dim  =  img_width  if  img_width  <  img_height  else  img_height
	crop_width  =  crop_height  =  crop_dim

	return  pil_img.crop(((img_width  -  crop_width)  //  2,  (img_height  -  crop_height)  //  2,  (img_width  +  crop_width)  //  2,  (img_height  +  crop_height)  //  2))
```

Once we've process this, we'll have `fg (100)`, `bg (100)`, `fg_mask (100)`, now we need to create the fg_bg images

Now to create the fg_bg images and also the fg_bg_mask images, we will place the fg images on top of bg images at random positions, 10 times, and do this with flipped fg images, in total we will have
`fg (100) x bg (100) x flip (2) x place_random (10) = fg_bg (400,000) + fg_bg_mask (400, 000)`

Code to do this,

```python
idx = 0
for bidx, bg_image in enumerate(tqdm(bgc_images)):

    if (bidx < last_idx):
        continue

	Path(f 'depth_dataset_cleaned/labels/').mkdir(parents = True, exist_ok = True)
	label_info = open(f "depth_dataset_cleaned/labels/bg_{bidx:03d}_label_info.txt", "w+")

	idx = 4000 * bidx

	print(f 'Processing BG {bidx}')
	Path(f 'depth_dataset_cleaned/fg_bg/bg_{bidx:03d}').mkdir(parents = True, exist_ok = True)
	Path(f 'depth_dataset_cleaned/fg_bg_mask/bg_{bidx:03d}').mkdir(parents = True, exist_ok = True)

	for fidx, fg_image in enumerate(tqdm(fgc_images)): #do the add fg to bg 20 times
	    for i in range(20): #do this twice, one with flip once without
	    for should_flip in [True, False]:
	    background = Image.open(bg_image)
	foreground = Image.open(fg_image)
	fg_mask = Image.open(fgc_mask_images[fidx])

	if should_flip:
	    foreground = foreground.transpose(PIL.Image.FLIP_LEFT_RIGHT)
	fg_mask = fg_mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

	b_width, b_height = background.size
	f_width, f_height = foreground.size
	max_y = b_height - f_height
	max_x = b_width - f_width
	pos_x = np.random.randint(low = 0, high = max_x, size = 1)[0]
	pos_y = np.random.randint(low = 0, high = max_y, size = 1)[0]
	background.paste(foreground, (pos_x, pos_y), foreground)

	mask_bg = Image.new('L', background.size)

	fg_mask = fg_mask.convert('L')
	mask_bg.paste(fg_mask, (pos_x, pos_y), fg_mask)

	background.save(f 'depth_dataset_cleaned/fg_bg/bg_{bidx:03d}/fg_{fidx:03d}_bg_{bidx:03d}_{idx:06d}.jpg', optimize = True, quality = 30)
	mask_bg.save(f 'depth_dataset_cleaned/fg_bg_mask/bg_{bidx:03d}/fg_{fidx:03d}_bg_{bidx:03d}_mask_{idx:06d}.jpg', optimize = True, quality = 30)
	label_info.write(f 'fg_bg/bg_{bidx:03d}/fg_{fidx:03d}_bg_{bidx:03d}_{idx:06d}.jpg\tfg_bg_mask/bg_{bidx:03d}/fg_{fidx:03d}_bg_{bidx:03d}_mask_{idx:06d}.jpg\t{pos_x}\t{pos_y}\n')

	idx = idx + 1

	label_info.close()
	last_idx = bidx
```

For efficiency i wrote the generated file to .zip file, why was this done though ?

[https://medium.com/@satyajitghana7/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477](https://medium.com/@satyajitghana7/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477)

Once this was done, we need to create the depth map, by running the DenseDepth Model on our fg_bg images, this was done by taking batches of 1000, since otherwise we had memory bottleneck issues, moreover i had to manually use the python's garbage collector to make sure we free the memory after every batch

```python
def run_processing(fr = 0, to = 10):
    print(f 'running process from {fr}(inclusive) to {to}(exclusive) BGs')
	for bdx, b_files in enumerate(tqdm(grouped_files[fr: to])):

	    print(f 'Processing for BG {fr + bdx}')

	out_zip = ZipFile('depth_fg_bg.zip', mode = 'a', compression = zipfile.ZIP_STORED)

	batch_size = 1000
	batch_idx = 0
	for batch in make_batch(b_files, batch_size):
	    images = []
	print(f 'Processing Batch {batch_idx}')
	for idx, b_file in enumerate(tqdm(batch)):
	    imgdata = fg_bg_zip.read(b_file)
	img = Image.open(io.BytesIO(imgdata))
	img = img.resize((640, 480))
	x = np.clip(np.asarray(img, dtype = float) / 255, 0, 1)
	images.append(x)

	images = np.stack(images, axis = 0)
	print(f 'Running prediction for BG {fr + bdx} Batch {batch_idx}')
	t1 = time()
	output = predict(model, images)
	outputs = output.copy()
	t2 = time()
	print(f 'Prediction done took {(t2-t1):.5f} s')

	# resize the outputs to `200x200` and extract channel 0
	outputs = [resize(output, (200, 200))[: ,: , 0]
	    for output in outputs
	]

	# create a temporary directory to save the png outputs of current bg directory
	Path(f 'temp_b').mkdir(parents = True, exist_ok = True)

	print('Saving to Zip File')# for every output, save the output by appending mask to it
	for odx, output in enumerate(tqdm(outputs)):
	    _, parent_f, f_name = b_files[batch_idx * batch_size + odx].split(os.sep)
	f_name = f_name.split('.')[0]
	img = Image.fromarray(output * 255)
	img = img.convert('L')
	img.save(f 'temp_b/temp.png')

	out_zip.write('temp_b/temp.png', f 'mask_fg_bg/{parent_f}/mask_{f_name}.png')

	# cleanup files
	del output, outputs, images

	# garbage collect
	gc.collect()

	batch_idx = batch_idx + 1

	out_zip.close()
```



--- 
dataset was made with 💖 by shadowleaf 😛
