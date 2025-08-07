I opened a folder named MapManagerCore

This is a git repo with a project to implement a Python backend for loading/displaying images and importantly for annotating images with customized regions of interest (ROI).

Can you examine this folder and give me an overview of what MapMAnagerCore is supposed to do?


- I am primarily interested in the backend architecture for ROI
- focus on code-level details


This code was written by an excellent programmer who has moved on to another job.

While I have an understanding of most of the code, I am stuck in one area.

You will see a class Spine which defines a schema to allow our lazy analysis of a pandas dataframe (akin to lazy loading).

There is one function addSchema() I am try to figure out.

For a single timepoint images, we have a number of color channels. Each ROI is shared across all color channels but each color channel computes values (like sum, mean, etc of pixels) in that channel.

Tell me what you know understand.

##

I guess my main question for you is, how would I improve the code such that if I have a single timepoint with a single channel (that mostly work), I then want to add another channel and expand the spine schema (and segment) to only then add in the lazy computations. This question is vague, feel free to help me out.

Second question, how could I extend this code to allow adding a new color channel but never have any roi calculation performed on it? Sometimes a 'color channel' is not raw data, an example would be a euclidean distance map.
