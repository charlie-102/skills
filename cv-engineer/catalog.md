# Computer Vision Engineer Skills Catalog

100 skills for Computer Vision Engineers. Request details on specific skills as needed.

**Also uses:** 25 shared skills from `_shared/catalog.md`

---

## 1. Image Fundamentals (10 skills)

| ID | Name | Description |
|----|------|-------------|
| CV01 | color-space-converter | Convert between RGB, HSV, LAB, YUV, XYZ color spaces |
| CV02 | image-format-handler | Read/write JPEG, PNG, TIFF, WebP, RAW, EXR formats |
| CV03 | resolution-manager | Resize, crop, pad images while preserving aspect ratio |
| CV04 | histogram-analyzer | Compute and visualize image histograms, apply equalization |
| CV05 | channel-operator | Split, merge, swap color channels, add alpha channels |
| CV06 | pixel-manipulator | Vectorized pixel-level operations (NumPy broadcasting) |
| CV07 | image-io-optimizer | Efficient batch loading with memory mapping and threading |
| CV08 | metadata-extractor | Read/write EXIF, ICC profiles, XMP metadata |
| CV09 | compression-tuner | Optimize JPEG quality, PNG compression, WebP settings |
| CV10 | batch-processor | Process image directories with multiprocessing/joblib |

## 2. Classical CV Techniques (12 skills)

| ID | Name | Description |
|----|------|-------------|
| CV11 | edge-detector | Apply Sobel, Canny, Laplacian, Scharr edge detection |
| CV12 | feature-extractor | Extract SIFT, SURF, ORB, AKAZE keypoints and descriptors |
| CV13 | morphology-operator | Erosion, dilation, opening, closing, gradient, top-hat |
| CV14 | contour-analyzer | Find contours, compute area, perimeter, convex hull, moments |
| CV15 | template-matcher | Template matching with NCC, SSD, multiple scales |
| CV16 | optical-flow-estimator | Dense (Farneback) and sparse (Lucas-Kanade) optical flow |
| CV17 | stereo-matcher | Stereo matching with BM, SGBM, disparity to depth |
| CV18 | image-registrator | Align images using feature matching or ECC |
| CV19 | keypoint-matcher | Match descriptors with BFMatcher, FLANN, ratio test |
| CV20 | ransac-verifier | RANSAC/MAGSAC for robust geometric estimation |
| CV21 | hough-detector | Detect lines (HoughLines) and circles (HoughCircles) |
| CV22 | corner-detector | Harris, Shi-Tomasi, FAST corner detection |

## 3. Image Filtering & Enhancement (10 skills)

| ID | Name | Description |
|----|------|-------------|
| CV23 | kernel-convolver | Apply custom convolution kernels (blur, edge, emboss) |
| CV24 | blur-operator | Gaussian, box, median, bilateral, motion blur |
| CV25 | sharpening-enhancer | Unsharp mask, Laplacian sharpening, high-pass filter |
| CV26 | noise-reducer | Non-local means, BM3D, Wiener filter for denoising |
| CV27 | contrast-adjuster | CLAHE, histogram stretching, gamma correction |
| CV28 | color-corrector | White balance, color grading, LUT application |
| CV29 | normalizer | Min-max, z-score, percentile normalization |
| CV30 | fft-filter | Frequency domain filtering (low-pass, high-pass, notch) |
| CV31 | deconvolver | Wiener, Richardson-Lucy deconvolution for deblurring |
| CV32 | hdr-processor | HDR merging, tone mapping (Reinhard, Drago, Mantiuk) |

## 4. Geometric Transformations (8 skills)

| ID | Name | Description |
|----|------|-------------|
| CV33 | affine-transformer | Rotation, scaling, shearing, translation with affine matrix |
| CV34 | perspective-warper | Homography estimation and perspective correction |
| CV35 | image-warper | Arbitrary warping with displacement fields |
| CV36 | interpolator | Bilinear, bicubic, Lanczos, area interpolation methods |
| CV37 | camera-calibrator | Intrinsic/extrinsic calibration with checkerboard patterns |
| CV38 | distortion-corrector | Radial and tangential lens distortion correction |
| CV39 | panorama-stitcher | Image stitching, blending, and seam finding |
| CV40 | coordinate-converter | Image to world coordinates, camera projection models |

## 5. Segmentation (10 skills)

| ID | Name | Description |
|----|------|-------------|
| CV41 | thresholding-operator | Otsu, adaptive, multi-level thresholding |
| CV42 | watershed-segmenter | Watershed algorithm with marker-based control |
| CV43 | grabcut-segmenter | Interactive foreground/background segmentation |
| CV44 | superpixel-generator | SLIC, SEEDS, LSC superpixel algorithms |
| CV45 | semantic-segmenter | Pixel-wise classification (DeepLabV3, UNet, SegFormer) |
| CV46 | instance-segmenter | Object instances with masks (Mask R-CNN, YOLACT) |
| CV47 | panoptic-segmenter | Combined stuff and things segmentation |
| CV48 | boundary-detector | Edge-aware boundary detection (HED, BDCN) |
| CV49 | region-grower | Region growing and split-merge algorithms |
| CV50 | background-subtractor | MOG2, KNN, GMM background subtraction for video |

## 6. Object Detection & Tracking (10 skills)

| ID | Name | Description |
|----|------|-------------|
| CV51 | bbox-operator | IoU, GIoU, DIoU calculation, box encoding/decoding |
| CV52 | nms-processor | Soft-NMS, DIoU-NMS, batched NMS, class-agnostic NMS |
| CV53 | multiscale-detector | FPN, feature pyramid, anchor pyramid handling |
| CV54 | object-tracker | SORT, DeepSORT, ByteTrack, OC-SORT implementation |
| CV55 | reid-extractor | Person/vehicle re-identification embeddings |
| CV56 | motion-detector | Frame differencing, MOG, motion heatmaps |
| CV57 | trajectory-analyzer | Track interpolation, smoothing, anomaly detection |
| CV58 | mot-handler | Multi-object tracking metrics (MOTA, IDF1, HOTA) |
| CV59 | occlusion-handler | Track through occlusions, re-identification |
| CV60 | detection-postprocessor | Score calibration, class balancing, ensemble boxes |

## 7. Image Restoration (10 skills)

| ID | Name | Description |
|----|------|-------------|
| CV61 | super-resolver | Classical (bicubic+) and deep SR (ESRGAN, SwinIR) |
| CV62 | image-denoiser | Gaussian, Poisson, salt-pepper noise removal |
| CV63 | deblurrer | Motion and out-of-focus deblurring |
| CV64 | inpainter | Content-aware fill, patch-based, deep inpainting |
| CV65 | artifact-remover | JPEG compression artifact and banding removal |
| CV66 | lowlight-enhancer | Low-light image/video enhancement (Zero-DCE, RetinexNet) |
| CV67 | dehaze-derain | Atmospheric degradation removal |
| CV68 | shadow-remover | Shadow detection and removal |
| CV69 | scratch-repairer | Old photo restoration, scratch and damage repair |
| CV70 | iqm-calculator | PSNR, SSIM, MS-SSIM, LPIPS, FID, NIQE metrics |

## 8. Face & Body Analysis (8 skills)

| ID | Name | Description |
|----|------|-------------|
| CV71 | face-detector | MTCNN, RetinaFace, BlazeFace, MediaPipe face detection |
| CV72 | face-recognizer | ArcFace, CosFace embedding extraction and matching |
| CV73 | landmark-detector | 68-point, 3D landmarks, MediaPipe face mesh |
| CV74 | face-aligner | 5-point alignment, 3D face alignment |
| CV75 | attribute-estimator | Age, gender, expression, head pose estimation |
| CV76 | pose-estimator | 2D/3D body pose (OpenPose, MediaPipe, DEKR) |
| CV77 | body-segmenter | Human parsing, body part segmentation |
| CV78 | action-recognizer | Video action recognition, gesture detection |

## 9. Video Processing (8 skills)

| ID | Name | Description |
|----|------|-------------|
| CV79 | frame-extractor | Video decoding, keyframe extraction, thumbnailing |
| CV80 | video-stabilizer | 2D/3D video stabilization, rolling shutter correction |
| CV81 | temporal-filter | Temporal smoothing, frame averaging, motion compensation |
| CV82 | scene-detector | Shot boundary detection, scene change analysis |
| CV83 | video-encoder | FFmpeg integration, codec selection, quality tuning |
| CV84 | frame-interpolator | RIFE, FILM, optical flow interpolation |
| CV85 | video-summarizer | Key frame selection, highlight detection |
| CV86 | realtime-pipeline | Low-latency video processing pipeline design |

## 10. 3D Vision (8 skills)

| ID | Name | Description |
|----|------|-------------|
| CV87 | depth-estimator | Monocular (MiDaS, DPT) and stereo depth estimation |
| CV88 | pointcloud-processor | PCL/Open3D operations, filtering, registration |
| CV89 | sfm-builder | Structure from Motion with COLMAP, OpenSfM |
| CV90 | slam-integrator | Visual SLAM basics (ORB-SLAM, RTAB-Map) |
| CV91 | mesh-reconstructor | Poisson, marching cubes, neural reconstruction |
| CV92 | nerf-renderer | NeRF training and rendering, instant-ngp |
| CV93 | multiview-geometry | Essential/fundamental matrix, triangulation |
| CV94 | 3d-detector | LiDAR and camera 3D object detection |

## 11. Domain-Specific CV (8 skills)

| ID | Name | Description |
|----|------|-------------|
| CV95 | medical-imager | CT/MRI/X-ray preprocessing, DICOM handling |
| CV96 | document-analyzer | OCR preprocessing, layout detection, table extraction |
| CV97 | driving-perceiver | Lane detection, sign recognition, ADAS features |
| CV98 | satellite-processor | GIS integration, orthorectification, change detection |
| CV99 | industrial-inspector | Defect detection, quality control imaging |
| CV100 | microscopy-analyzer | Cell segmentation, particle analysis, focus stacking |

## 12. CV Tools & Libraries (Bonus)

These are library-specific expertise areas integrated into the skills above:

| Tool | Primary Skills |
|------|----------------|
| OpenCV | CV11-CV22, CV23-CV32, CV33-CV40, CV41-CV50 |
| PIL/Pillow | CV01-CV10, CV23-CV32 |
| scikit-image | CV11-CV22, CV41-CV50, CV23-CV32 |
| albumentations | CV33-CV40, CV23-CV32 (augmentation focus) |
| TensorRT | CV45-CV47, CV51-CV60 (inference optimization) |
| ONNX Runtime | CV45-CV47, CV51-CV60 (deployment) |
| FFmpeg | CV79-CV86 (video processing) |
| Open3D/PCL | CV87-CV94 (3D vision) |

---

## Summary

| Category | Count | IDs |
|----------|-------|-----|
| Image Fundamentals | 10 | CV01-CV10 |
| Classical CV | 12 | CV11-CV22 |
| Filtering & Enhancement | 10 | CV23-CV32 |
| Geometric Transforms | 8 | CV33-CV40 |
| Segmentation | 10 | CV41-CV50 |
| Detection & Tracking | 10 | CV51-CV60 |
| Image Restoration | 10 | CV61-CV70 |
| Face & Body | 8 | CV71-CV78 |
| Video Processing | 8 | CV79-CV86 |
| 3D Vision | 8 | CV87-CV94 |
| Domain-Specific | 6 | CV95-CV100 |
| **Total** | **100** | |

---

## Request Details

To get detailed implementation for any skill, ask:
> "Give me details on CV54 (object-tracker)"

Or multiple:
> "Details on CV61, CV62, CV70 (restoration skills)"

---

## Key Differences from ML Engineer

| CV Engineer Focus | ML Engineer Focus |
|-------------------|-------------------|
| Image/video processing | General model training |
| Classical CV algorithms | MLOps pipelines |
| Pixel-level operations | Distributed training |
| Domain-specific vision | Multi-domain ML |
| OpenCV/FFmpeg workflows | PyTorch/TensorFlow workflows |
| Real-time pipelines | Batch processing |
| Visual quality metrics (PSNR, SSIM) | ML metrics (accuracy, F1) |
