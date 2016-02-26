# Description

This is a modified version of a forked [densecrf](http://www.philkr.net/home/densecrf), 
which was used as a part of the [DeepLab](https://bitbucket.org/deeplab/deeplab-public/).

For more details about the inference algorithm used in this version, please refer to and 
consider citing the following paper:
```
@article{baque2015principled,
  title={Principled Parallel Mean-Field Inference for Discrete Random Fields},
  author={Baqu{\'e}, Pierre and Bagautdinov, Timur and Fleuret, Fran{\c{c}}ois and Fua, Pascal},
  journal={arXiv preprint arXiv:1511.06103},
  year={2015}
}
```

If you are using densecrf, please consider citing the following paper:
```
@inproceedings{KrahenbuhlK11,
  title={Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials},
  author={Philipp Kr{\"{a}}henb{\"{u}}hl and Vladlen Koltun},
  booktitle={NIPS},      
  year={2011}
}
```

If you are using DeepLab, please consider citing following paper:
```
@article{papandreou15weak,
  title={Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation},
  author={George Papandreou and Liang-Chieh Chen and Kevin Murphy and Alan L Yuille},
  journal={arxiv:1502.02734},
  year={2015}
}
```


# Building and Dependencies

You should have [matio](https://sourceforge.net/projects/matio/) library installed.

To build the binary, just run `make`.

# Usage

... to be filled in ...

For the complete pipeline for semantic segmentation, please refer to 
[DeepLab](https://bitbucket.org/deeplab/deeplab-public/).

For the details (parameters) specific to this version, refer to 
`refine_pascal_nat/dense_inference.cpp`.


