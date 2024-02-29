## Amygdala

### Input and Output

**Actually maybe for both the input and output it should only be 27k instead of 30k because of the validation**

- Input x: (30k, 512) --> vector that matches the ordering of the activations
  - I need to construct a new x where i loop over the ordering and concatenate correct clip vectors together
  - This is because [ordering] is the order of the images shown to the subjects while the order in [clip_vectors] is the order of the images in the dataset
    - clip vectors should be in order of activation, not 0 to 9999 (order of the images in dataset)
- Output y: (30k, 241) 

### Train Mask

- each subject sees the images 3 times (30k trials)
- The first 1000 images were shown to all subjects so it is used as a validation set
  - so we only want to use images with ordering >= 1000
  - if you take those out we have 27000 left because 1000*3=3000
  - we need to do some sort of mask for training
  - apply the mask to both the clip vector and the activation

### Ridge Regression

- I want to use ridge regression to predict the activation from the clip vector
  - start for pretty small alpha and see how the performance changes (0.01, 0.1, etc)
  - use score() --> X should be validation x, y is validation y, and weight is the weights we just trained
  - get_score() should give me the R^2
  - Haomiao expects R<0.1 like. 0.06, 0.07
  - Then, R^2 will be like 0.003