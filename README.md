Download Link: https://assignmentchef.com/product/solved-cv1-lab-2-neighborhood-processing-and-filters
<br>
In this assignment, you will get familiar with fundamentals of neighborhood processing for image processing. These techniques allow for low-level image understanding via extraction of structural patterns such as edges and blobs. Similarly, they find an extensive use in image denoising and higher level image reasoning such as shape recognition. Moreover, neighborhood or block processing is one of the key components of <em>Convolutional Neural Networks</em>. Therefore, a good understanding of these procedures will be a stepping stone towards understanding more complex machinery used in computer vision and machine learning.

In subsequent sections of this assignment, we will first explain neighborhood processing and introduce low-level filters commonly used to analyze images. After that, we will see how these mathematical concepts relate to practice by working through fundamental tasks such as denoising and segmentation. By the end of this assignment, you will have an overall understanding of the following:

<ul>

 <li>Gaussian and Gabor filters</li>

 <li>Edge detection and image denoising</li>

 <li>Texture-based image segmentation</li>

</ul>

<h1><a name="_Toc25552"></a>1             Neighborhood Processing</h1>

Neighborhood processing is simply about looking around a point <strong>I</strong>(<em>x,y</em>) (i.e. pixel) in the image, <strong>I</strong>, and applying a function, <strong>h</strong>(<em>k,l</em>), which measures certain properties or relationships between the pixels in that localized window. The function, <strong>h</strong>(<em>k,l</em>), is generally referred to as the <em>neighborhood </em>operator or <em>local </em>operator. One of the most common forms of a neighborhood operator is a <em>linear filter</em>. Linear filters simply compute the weighted sum of neighboring pixel intensities and assign it to the pixel of interest (output <strong>I</strong><em><sub>out</sub></em>(<em>i,j</em>)). The filters in which we are interested here are usually represented as a square matrix.

<strong>Hint</strong>

Filters, kernels, weight matrices or masks are interchangeably used in the literature. A kernel is a matrix with which we describe a neighborhood operation. This operation can, for example, be edge detection or smoothing.

Linear filters are shifted over the entire image plane via operators such as correlation (⊗) and convolution (∗). Both of these operators are <em>linear shift-invariant </em>(LSI) implying that the filters behave the same way over the entire image. Discrete forms of these operators are given in the following:

We illustrate the overall idea of neighborhood processing in Figure 1.

Figure 1: The kernel or the mask convolves over the input image. In the case of linear filters, this is simply multiplying each pixel intensity with the corresponding weight in the kernel (see the yellowish 7×7 window where the kernel is placed). In the figure, the kernel is 7×7 averaging mask. You can see its effect by comparing the red (before filtering) and the green (after filtering) frames.

<h1><a name="_Toc25553"></a>2             Low-level filters</h1>

In this section, you will design common linear filters used in neighborhood processing. We will focus in particular on Gaussian and Gabor filters.

<h2><a name="_Toc25554"></a>2.1           Gaussian Filters</h2>

<h3><a name="_Toc25555"></a>2.1.1          1D Gaussian Filter</h3>

The 1D Gaussian filter is defined as follows:

<em>,                                           </em>(3)

where <em>σ </em>is the variance of the Gaussian. However, such formulation creates an infinitely large convolution kernel. In practice, the kernel is truncated with a kernel_size parameter such that , where b<em>.</em>c is the floor operator. As an example, if kernel_size equals 3, <em>x </em>∈ {−1<em>,</em>0<em>,</em>1}.

<h3><a name="_Toc25556"></a>2.1.2          2D Gaussian Filter</h3>

One of the most important properties of 2D Gaussian kernels is separability. Therefore, convolving an image with a 2D Gaussian is equivalent to convolving the image twice with a 1D Gaussian filter, once along the x-axis and once along the y-axis <strong>separately</strong>. A 2D Gaussian kernel can then be defined as a product of two 1D

Gaussian kernels:

)                                                 (4)

)                                      (5)

<h3><a name="_Toc25557"></a>2.1.3          Gaussian Derivatives</h3>

So far the Gaussian kernels that we computed are mainly targeted to image enhancement algorithms (e.g. denoising an image). These kernels can also be used for detecting changes in the image intensity pixels. These low-level features can then further be used as building blocks for more complicated tasks like object detection or segmentation.

Concretely, the first order derivative of the 1D Gaussian kernel is given by:

(6)

Similarly, the first order derivative of the 2D Gaussian kernel can be obtained by computing ) and

<h2><a name="_Toc25558"></a>2.2           Gabor filters</h2>

Gabor filters fall into the category of linear filters and are widely used for <em>texture analysis</em>. The reason why they are a good choice for texture analysis is that they localize well in the frequency spectrum (<em>optimally </em>bandlimited) and therefore work as flexible <em>band-pass </em>filters. See Figures 2 and 3.

<h3><a name="_Toc25559"></a>2.2.1          1D Gabor Filters</h3>

For the sake of simplicity, we start by studying what a Gabor function is using 1D signals (e.g. speech). The idea will later be generalized to the 2D case, which is suited for our primary interest, images.

A Gabor function is a Gaussian function modulated with a complex sinusoidal carrier signal. Let us denote the Gaussian with <em>x</em>(<em>t</em>) and complex sinusoidal with <em>m</em>(<em>t</em>). Then, a Gabor function <em>g</em>(<em>t</em>) can be formulated by

<em>g</em>(<em>t</em>) = <em>x</em>(<em>t</em>)<em>m</em>(<em>t</em>)                                                      (7)

where and <em>m</em>(<em>t</em>) = <em>e</em><sup>−<em>j</em>2<em>πf</em></sup><em><sup>ct </sup></em>= <em>e</em><sup>−<em>jw</em></sup><em><sup>ct</sup></em>. <em>σ </em>is the parameter determining the spread of the Gaussian and <em>w<sub>c </sub></em>is the central frequency of the carrier signal.

Using Euler’s formula, we get the following:

)                                                                          (8)

(9)

)]                             (10)

<table width="567">

 <tbody>

  <tr>

   <td width="539">We can further arrange the terms and arrive at the following form</td>

   <td width="28"> </td>

  </tr>

  <tr>

   <td width="539"><em>g</em>(<em>t</em>) = <em>g<sub>e</sub></em>(<em>t</em>) + <em>jg<sub>o</sub></em>(<em>t</em>)</td>

   <td width="28">(11)</td>

  </tr>

 </tbody>

</table>

where <em>g<sub>e</sub></em>(<em>t</em>) and <em>g<sub>o</sub></em>(<em>t</em>) are the even and odd parts arranged orthogonally on the complex plane <strong>Z</strong><sup>2</sup>. In practice, one can use either the even or the odd part for filtering purposes (or one can use the complex form).

<h3><a name="_Toc25560"></a>2.2.2          2D Gabor Filters</h3>

The Gabor filters can also be defined in 2D as well. The main difference lies in the dimensionality of the signals (i.e. carrier and gaussian). A sine wave in 2D is described by two orthogonal spatial frequencies <em>u</em><sub>0 </sub>and <em>v</em><sub>0 </sub>such that it is given as

<em>s</em>(<em>x,y</em>) = <em>sin</em>(2<em>π</em>(<em>u</em><sub>0</sub><em>x</em>+<em>v</em><sub>0</sub><em>y</em>)) where a 2D gaussian is simply with

<em>C </em>being a normalizing constant. 2D Gabor function then takes the following forms in the real and complex parts:

(12)

<em>,           </em>(13)

Figure 2: Even (cosine-modulated) and odd parts (sine-modulated) of Gabor filters with fixed-<em>σ </em>Gaussian. We illustrate the time-domain filters for the modulating sinusoidals of central frequencies, 10, 20, 30, 40 and 50 Hz, respectively.

Figure 3: Gabor filters with varying center frequencies are sensitive to different frequency bands. Notice that the neighboring (in the frequency spectrum) filters minimally interfere with each other.

where

<table width="358">

 <tbody>

  <tr>

   <td width="330"><em>x</em><sup>0 </sup>= <em>x</em>cos<em>θ </em>+ <em>y </em>sin<em>θ</em></td>

   <td width="28">(14)</td>

  </tr>

  <tr>

   <td width="330"><em>y</em><sup>0 </sup>= −<em>x</em>sin<em>θ </em>+ <em>y </em>cos<em>θ</em></td>

   <td width="28">(15)</td>

  </tr>

 </tbody>

</table>

<h1><a name="_Toc25561"></a>3             Applications in image processing</h1>

<h2><a name="_Toc25562"></a>3.1           Noise in digital images</h2>

The quality of digital images can be affected in different ways. For example, the acquisition process can be very noisy and with a low-resolution (e.g. some medical imaging modalities only generate a 128×128 image). Noise can also come from the user who set wrong parameters on the digital camera. Consequently, different computer vision algorithms are required to enhance noisy or corrupted images. With the growing amount of photos taken every day, image enhancement has then become a very active area of research.

In this section, we only focus on simple algorithms to correct noise coming typically from the sensor of your camera. Many other types of noise or corruption can happen but are out of the scope of this assignment.

<h3><a name="_Toc25563"></a>3.1.1          Salt-and-pepper noise</h3>

Noise can also occur with over-exposition causing a ”hot” pixel or with a defective sensor causing a ”dead” pixel. This is called salt-and-pepper noise. Pixels in the image are randomly replaced by either a white or black pixel.

<h3><a name="_Toc25564"></a>3.1.2          Additive Gaussian noise</h3>

Noise also occurs frequently when the camera heats up. This is called thermal noise and this can be modeled as an additive Gaussian noise. Every pixel in the image has a noise component that corresponds to a random value chosen independently from the same Gaussian probability distribution. The Gaussian distribution has a mean of 0 and its standard deviation corresponds to a parameter.

<strong>I</strong>, where                                 (16)

where <strong>I</strong><sup>0 </sup>is the noisy image and <strong>I </strong>is the original image without any noise .

<h2><a name="_Toc25565"></a>3.2           Image denoising</h2>

<h3><a name="_Toc25566"></a>3.2.1          Quantitative evaluation</h3>

The peak signal-to-noise ratio (PSNR) is a commonly used metric to quantitatively evaluate the performance of image enhancement algorithms. It is derived from the mean squared error (MSE):

(17)

where <strong>I </strong>is the original image of size <em>m </em>× <em>n </em>and <strong>ˆI </strong>its approximation (i.e. in our case an enhanced corrupted image). The PSNR corresponds to:

(18)

where <strong>I</strong><em><sub>max </sub></em>is the maximum pixel value of <strong>I </strong>and RMSE is the root of the MSE.

<h3><a name="_Toc25567"></a>3.2.2          Neighborhood processing for image denoising</h3>

We will now design filters to remove these two types of noise. The function will denoise the image by either applying:

<ol>

 <li><em>box filtering</em>: You can use blur function.</li>

 <li><em>median filtering</em>: You can use medianBlur function.</li>

 <li><em>Gaussian filtering</em>: You must use your GaussianBlur function. <strong>Implement </strong>denoise</li>

</ol>

def denoise(image, kernel_type, **kwargs):

…

return imOut

<h4>Hints</h4>

<ol>

 <li>kernel_type is just a string to specify the kernel type.</li>

 <li>**kwargs allows to have an undefined key-value pairs in a Python function. For example, you can have sigma and kernel_size as argument when using a Gaussian kernel but only kernel_size when using a box kernel. For more information about how **kwargs works, take a look at <a href="https://book.pythontips.com/en/latest/args_and_kwargs.html#usage-of-kwargs">usage of kwargs.</a></li>

</ol>

<h4>Question 7 (20<em>pts</em>)</h4>

<ol>

 <li>Using your    implemented  function <strong>denoise</strong>,     try       denoising       image1 saltpepper.jpg and image1 gaussian.jpg by applying the following filters:

  <ul>

   <li>Box filtering of size: 3×3, 5×5, and 7×7.</li>

   <li>Median filtering with size: 3×3, 5×5 and 7×7.</li>

  </ul></li>

</ol>

Show the denoised images in your report. Use tables to present your quantitative results.

<ol start="2">

 <li>Using your implemented function <strong>myPSNR</strong>, compute the PSNR for every denoised image (12 in total) wrt the original image. What is the effect of the filter size on the PSNR? Report the results in a table and discuss.</li>

 <li>Which is better for the salt-and-pepper noise, box or median filters? Why? What about the Gaussian noise?</li>

 <li>Try denoising image1 gaussian.jpg using a Gaussian filtering. Choose an appropriate window size and standard deviation and justify your choice. Show the denoised images in your report.</li>

 <li>What is the effect of the standard deviation on the PSNR? Report the results in a table and discuss.</li>

 <li>What is the difference among median filtering, box filtering and Gaussian filtering? Briefly explain how they are different at a conceptual level. If two filtering methods give a PSNR in the same ballpark, can you see a qualitative difference?</li>

</ol>

<h2><a name="_Toc25568"></a>3.3           Edge detection</h2>

Edges appear when there is a sharp change in brightness. In an image this usually corresponds to the boundaries of an object. Edge detection is a fundamental task used in many computer vision applications. One of them is road detection in autonomous driving, which is used for determining the vehicle trajectory.

Many different techniques exist for computing the edges. In this section, we will focus on filters that extract the gradient of the image. We will try to detect the road in an still image.

<h3><a name="_Toc25569"></a>3.3.1          First-order derivative filters</h3>

<strong>Sobel </strong>kernels approximate the first derivative of a Gaussian filter. Below are the Sobel kernels used in the <em>x </em>and <em>y </em>directions.

<strong>I                                             </strong>(19)

<strong>I                                           </strong>(20)

The gradient magnitude is defined as the square root of the sum of the squares of the horizontal (<em>G<sub>x</sub></em>) and the vertical (<em>G<sub>y</sub></em>) components of the gradient of an image, such that:

q

<em>G </em>=        <em>G<sub>x</sub></em><sup>2 </sup>+ <em>G<sub>y</sub></em><sup>2                                                                                                     </sup>(21)

The gradient direction is calculated as follows:

(22)

<h3><a name="_Toc25570"></a>3.3.2          Second-order derivative filters</h3>

Compared to the Sobel filter, a Laplacian of Gaussian (LoG) relies on the second derivative of a Gaussian filter. Hence, it will focus on large gradients in the image. A LoG can be computed by the following three methods:

<ul>

 <li>method 1: Smoothing the image with a Gaussian kernel (kernel size of 5 and standard deviation of 0.5), then taking the Laplacian of the smoothed image (i.e. second derivative).</li>

 <li>method 2: Convolving the image directly with a LoG kernel (kernel size of 5 and standard deviation of 0.5).</li>

 <li>method 3: Taking the Difference of two Gaussians (DoG) computed at different scales <em>σ</em><sub>1 </sub>and <em>σ</em><sub>2</sub>.</li>

</ul>

<strong>Implement </strong>compute_LoG

The function should be able to apply any of the above mentioned methods depending on the value passed to the parameter <em>LOG type</em>.

def compute_LoG(image, LOG_type):

… return imOut

<em>Note: </em>You are not allowed to use the Python built-in functions for computing LOG kernels. But for doing 2D convolution, you can benefit from <em>scipy.signal.convolve2d </em>function.

<h4>Question 9 (10<em>pts</em>)</h4>

<ol>

 <li>Test your function using image2.jpg and visualize your results using the three methods.</li>

 <li>Discuss the difference between applying the three methods.</li>

 <li>In the first method, why is it important to convolve an image with a Gaussian before convolving with a Laplacian?</li>

 <li>In the third method, what is the best ratio between <em>σ</em><sub>1 </sub>and <em>σ</em><sub>2 </sub>to achieve the best approximation of the LoG? What is the purpose of having 2 standard deviations?</li>

 <li>What else is needed to improve the performance and isolate the road, i.e. what else should be done? You don’t have to provide any specific parameter or specific algorithm. Try to propose a direction which would be interesting to explore and how you would approach it.</li>

</ol>

<h2><a name="_Toc25571"></a>3.4           Foreground-background separation</h2>

Foreground-background separation is an important task in the field of computer vision (see Figure 4). In this exercise, you will implement a simple unsupervised algorithm that leverages the variations in texture to segment the foreground object from the background. We will assume the foreground object has a distinct combination of textures compared to background. As mentioned earlier, Gabor filters are well-suited for texture analysis thanks to their frequency domain characteristics. Therefore, we will use a collection of Gabor filters with varying scale and orientations which we call a <em>filter bank</em>. The outline of the algorithm is as follows:

Figure 4: <strong>(Left) </strong>Input image, <strong>(Middle) </strong>Foreground mask, <strong>(Right) </strong>Masked object. Foreground-Background separation aims at masking out the salient object pixels from the background pixels.

<strong>Algorithm 1 </strong>Foreground-Background Segmentation Algorithm

<strong>Input: </strong><em>x </em>– input image

<strong>Output: </strong><em>y </em>– pixelwise labels

<ol>

 <li>Convert to grayscale if necessary. <strong>if </strong><em>x </em>is RGB <strong>then</strong></li>

</ol>

<em>x </em>← rgb2gray(<em>x</em>)

<h4>end if</h4>

<ol start="2">

 <li>Create Gabor filterbank, F<em><sub>gabor</sub></em>, with varying <em>σ</em>, <em>λ </em>and <em>θ</em>.</li>

 <li>Filter <em>x </em>with the filterbank. Store each output in <em>fmaps</em>. <em>fmaps </em>← F<em><sub>gabor</sub></em>(<em>x</em>)</li>

 <li>Compute the magnitude of the complex <em>fmaps</em>. Store the results in <em>fmags</em>.</li>

</ol>

<em>fmags </em>← |<em>fmaps</em>|

<ol start="5">

 <li>Smooth <em>fmags</em>. <em>fmags </em>← smooth(<em>fmags</em>)</li>

 <li>Convert <em>fmags </em>into data matrix, <em>f</em>. <em>f </em>← reshape(<em>fmags</em>)</li>

 <li>Cluster <em>f </em>using kmeans into two sets.</li>

</ol>

<em>y </em>← kmeans(<em>f</em>, 2)

We provide you with additional instructions in the <strong>gabor segmentation.py </strong>file.

<strong>Implement </strong>gabor_segmentation

Please get yourself familiar with provided skeleton code <strong>gabor segmentation.py</strong>. Keep in mind that you will need your implementation of the <strong>createGabor </strong>function.

Implement all code sections where you see a comment in the form:

# \TODO: xxx

When you succesfully implement it all, it should run without problems and produce a reasonable segmentation with the default parameters on <em>kobi.png</em>.

<h5>Question 10 (20<em>pts</em>)</h5>

<ol>

 <li>Run the algorithm on all test images with the provided parameter settings. What do you observe? Explain shortly in the report.</li>

 <li>Experiment with different <em>λ</em>, <em>σ </em>and <em>θ </em>settings until you get reasonable outputs. Report what parameter settings work better for each input image and try to explain why.</li>

</ol>

<em>Hint: </em>Don’t change multiple variables at once. You might not need to change some at all.

<ol start="3">

 <li>After you achieve good separation on all test images, run the script again with corresponding parameters but this time with smoothingFlag = False</li>

</ol>

Describe what you observe at the output when smoothing is not applied on the magnitude images. Explain why it happens and try to reason about the motivation behind this step.