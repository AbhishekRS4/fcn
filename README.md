# FCN implementation on Cityscapes dataset

## Notes
* Implementations of FCN-8, FCN-16 and FCN-32 with VGG-16 as the base
* Images resized to 1024x512 are used to train the model
* 15 custom classes used

## Instructions to run
* To list training options
```
python3 fcn_train.py --help
```
* To list inference options
```
python3 fcn_infer.py --help
```

## Visualization of results
* [FCN-8](https://youtu.be/LF6KWpqYUkE)
* [FCN-16](https://youtu.be/BcNDZuUEgnM)
* [FCN-32](https://youtu.be/WYw5gF1FH70)

## Reference
* [VGG](https://arxiv.org/abs/1409.1556)
* [FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
