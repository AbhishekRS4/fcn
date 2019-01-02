# FCN implementation on Cityscapes dataset

## Notes
* Implementations of FCN-8, FCN-16 and FCN-32 with VGG-16 as the base
* Images resized to 1024x512 are used to train the model
* 15 custom classes used

## Instructions to run
> To run training use - **python3 fcn\_train.py -h**
>
> To run inference use - **python3 fcn\_infer.py -h**
>
> This lists all possible commandline arguments

## Visualization of results
* [FCN-8](https://youtu.be/LF6KWpqYUkE)
* [FCN-16](https://youtu.be/BcNDZuUEgnM)
* [FCN-32](https://youtu.be/WYw5gF1FH70)

## Reference
* [VGG](https://arxiv.org/abs/1409.1556)
* [FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

## To do
- [x] load pretrained vgg-16 model
- [x] fcn-8
- [x] fcn-16
- [x] fcn-32
- [x] visualize results 
- [ ] compute metrics for validation set
