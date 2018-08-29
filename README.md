# Integrated Gradients
(*a.k.a. Path-Integrated Gradients, a.k.a. Axiomatic Attribution for Deep
Networks*)

**Contact**: integrated-gradients AT gmail.com

**Contributors** (alphabetical, last name):
* Kedar Dhamdhere (Google)
* Pramod Kaushik Mudrakarta (U. Chicago)
* Mukund Sundararajan (Google)
* Ankur Taly (Google Brain)
* Jinhua (Shawn) Xu  (Verily)

We study the problem of attributing the prediction of a deep network to its
input features, as an attempt towards explaining individual predictions. For
instance, in an object recognition network, an attribution method could tell us
which pixels of the image were responsible for a certain label being picked, or
which words from sentence were indicative of strong sentiment.

Applications range from helping a developer debug, allowing analysts to explore
the logic of a network, and to give end-user’s some transparency into the reason
for a network’s prediction.

**Integrated Gradients** is a variation on computing the gradient of the
prediction output w.r.t. features of the input. It requires no modification to
the original network, is simple to implement, and is applicable to a variety of
deep models (sparse and dense, text and vision).

## Relevant papers and slide decks

* [**Axiomatic Attribution for Deep Networks**][icml-paper] -- *Mukund Sundararajan, Ankur Taly, Qiqi Yan*, Proceedings of International Conference on Machine Learning (**ICML**), 2017
  
  This paper introduced the Integrated Gradients method. It presents an axiomatic justification of the method along with applications to various deep networks.
  [Slide deck][icml-slides]
  
* [**Did the model understand the questions?**][acl-paper] -- *Pramod Mudrakarta, Ankur Taly, Mukund Sundararajan, Kedar Dhamdhere*, Proceedings of Association of Computational Linguistics (**ACL**), 2018
  
  This paper discusses an application of integrated gradients for evaluating the robustness of question-answering networks. 
  [Slide deck][acl-slides]


## Implementing Integrated Gradients 

This [How-To document][howto] describes the steps involved in implementing integrated gradients
for an arbitrary deep network.

This repository provideds code for implementing integrated gradients for networks
with image inputs. It is structured as follows:
* [Integrated Gradients library][intgrad-lib]: Library implementing the core
integrated gradients algorithm.
* [Visualization library][vis-lib]: Library implementing methods for visualizing
atributions for image models.
* [Inception notebook][incp-notebook]: A [Jupyter](http://jupyter.org/) notebook
for generating and visualizing atributions for the [Inception (v1)][incp-paper]
object recognition network.

We recommend starting with the notebook. To run the notebook, please follow the following instructions.
* Clone this repository
  
  ```
  git clone https://github.com/ankurtaly/Attributions
  ```
* In the same directory, run the Jupyter notebook server.
  
  ```
  jupyter notebook
  ```
  Instructions for installing Jupyter are available [here](http://jupyter.readthedocs.io/en/latest/install.html).
  Please make sure that you have [TensorFlow][tf], [NumPy][np], and [PIL.Image][pillow] installed for
  Python 2.7.
* Open `attributions.ipynb` and run all cells.

<!---
## Visualizations

Below are some visualizations of interior gradients (as a GIF) and integrated
gradients for some images from the ImageNet object recognition dataset. For
comparison, we also show a visualization of the gradients at the actual image.

### Image: 6864d7789068273e
![6864d7789068273e](/Visualizations/IntegratedGradients/6864d7789068273e.jpg)
### Image: e60dfdf262c5c74f
![e60dfdf262c5c74f](/Visualizations/IntegratedGradients/e60dfdf262c5c74f.jpg)
### Image: 35655d94a4557fbb
![35655d94a4557fbb](/Visualizations/IntegratedGradients/35655d94a4557fbb.jpg)
### Image: bda7f59b986d42c0
![bda7f59b986d42c0](/Visualizations/IntegratedGradients/bda7f59b986d42c0.jpg)
### Image: 96bb6e948866e5b5
![96bb6e948866e5b5](/Visualizations/IntegratedGradients/96bb6e948866e5b5.jpg)
### Image: 80c64f2e27f8784a
![80c64f2e27f8784a](/Visualizations/IntegratedGradients/80c64f2e27f8784a.jpg)
### :gImage: b19f875f181025d3
![b19f875f181025d3](/Visualizations/IntegratedGradients/b19f875f181025d3.jpg)
### Image: 2f401b18be02d7fe
![2f401b18be02d7fe](/Visualizations/IntegratedGradients/2f401b18be02d7fe.jpg)
### Image: 92445d6529368418
![92445d6529368418](/Visualizations/IntegratedGradients/92445d6529368418.jpg)
### Image: ba1011c8f9633a49
![ba1011c8f9633a49](/Visualizations/IntegratedGradients/ba1011c8f9633a49.jpg)
### Image: 3721d654514bc885
![3721d654514bc885](/Visualizations/IntegratedGradients/3721d654514bc885.jpg)
### Image: 12ec21fa7003eae5
![12ec21fa7003eae5](/Visualizations/IntegratedGradients/12ec21fa7003eae5.jpg)
### Image: 82262660db12ad85
![82262660db12ad85](/Visualizations/IntegratedGradients/82262660db12ad85.jpg)
### Image: b2ab69fbb052b435
![b2ab69fbb052b435](/Visualizations/IntegratedGradients/b2ab69fbb052b435.jpg)
### Image: 9a7e268c95022a1c
![9a7e268c95022a1c](/Visualizations/IntegratedGradients/9a7e268c95022a1c.jpg)
### Image: 023d8b91c64faf4b
![023d8b91c64faf4b](/Visualizations/IntegratedGradients/023d8b91c64faf4b.jpg)
### Image: 7f12674c6943381d
![7f12674c6943381d](/Visualizations/IntegratedGradients/7f12674c6943381d.jpg)
### Image: 7bcfe0265ad6a2b5
![7bcfe0265ad6a2b5](/Visualizations/IntegratedGradients/7bcfe0265ad6a2b5.jpg)
### Image: 518a1c0660c5e32e
![518a1c0660c5e32e](/Visualizations/IntegratedGradients/518a1c0660c5e32e.jpg)
### Image: e9c7c07cb5730dac
![e9c7c07cb5730dac](/Visualizations/IntegratedGradients/e9c7c07cb5730dac.jpg)
-->

[howto]:https://github.com/ankurtaly/Integrated-Gradients/tree/master/howto.md
[intgrad-lib]:https://github.com/ankurtaly/Integrated-Gradients/tree/master/IntegratedGradients
[vis-lib]:https://github.com/ankurtaly/Integrated-Gradients/tree/master/VisualizationLibrary
[incp-notebook]:https://github.com/ankurtaly/Integrated-Gradients/blob/master/attributions.ipynb
[incp-paper]:http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf
[icml-paper]:http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf
[acl-paper]:https://arxiv.org/abs/1805.05492
[icml-slides]: https://github.com/ankurtaly/Integrated-Gradients/tree/master/icml_slides.pdf
[acl-slides]:https://github.com/pramodkaushik/acl18_results/blob/master/talk_slides_ACL2018.pdf
[attributions-code]:https://github.com/ankurtaly/Attributions/blob/master/attributions.ipynb
[tf]:https://www.tensorflow.org/install/
[np]:https://docs.scipy.org/doc/numpy/user/install.html
[pillow]:http://pillow.readthedocs.io/en/3.1.x/installation.html
