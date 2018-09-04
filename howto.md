# How to use Integrated Gradients (IG)

To use this method, you'll need to:
*   Identify the [input](#identifying-the-input-tensor) and [output](#identifying-the-output-tensor)
*   Select a [baseline](#why-do-you-need-a-baseline) to diff the input against
*   Select the [number of steps](#tuning-the-number-of-steps-in-the-gradient-approximation) in the integral approximation
*   Run [sanity checks](#sanity-checking-baselines)
*   [Visualize](#visualizing-the-attributions) the results

Reference implementation:
*   [Python Notebook for Inception model](https://github.com/ankurtaly/Integrated-Gradients/blob/master/attributions.ipynb) (object recognition model for images)


## What are the capabilities and limitations of IG

Integrated Gradients is a systematic technique that attributes a deep model's prediction to its base features. For instance, an object recognition network's prediction to its pixels or a sentiment model's prediction to individual words in the sentence.The technique is based on the [paper](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) at ICML'17, a top tier machine learning conference.

[Variants ](https://arxiv.org/abs/1805.12233)of IG can be used to apply the notion of attribution to neurons.

That said, IG does not uncover the logic used by the network to combine features, though there are variants of IG that can do this in a limited sense. 


## Identifying the output tensor

*   The attributions must be from the prediction head of the deep learning model.  For models with a sequence output, attributions must be computed separately for each predicted token in the output sequence.
*   For multi-class classification models, the prediction head is typically a softmax operator on a 'logits' tensor. The attribution must be computed from this softmax output and not the 'logits' tensor.
*   **Sanity check**: Ensure that the tensor corresponds to a single prediction label. This can be done by checking that the shape of the tensor is of the form <batch,> (instead of <batch, num_labels>)


## Identifying the input tensor
*   For models with dense input, e.g., images or speech, attribute directly to the base input layer. 
*   Models with sparse input, e.g., text, first embed the input into a dense tensor, which is  then fed to multi-layer network. Attribution must be performed to this embedding tensor, but before the token embeddings are combined into a single input embedding. 
*   **Implementation Notes**: Depending on how the network is implemented, getting hold of the embedding tensor might be tricky. If the implementation uses an explicit embedding lookup operation (e.g., tf.nn.embedding_lookup) then we could grep for the operation in the graph and note its output tensor. 

    On the other hand the implementation may use [tf.feature_column API](https://www.tensorflow.org/guide/feature_columns) to directly embed raw feature values to continuous embedding tensors. In such cases there are a couple of options:

    *   Traverse the graph backwards and get hold of the embedding operation. 
    *   The recommended pattern for combining embedded feature columns into dense  tensors is to use [tf.feature_column.input_layer](https://www.tensorflow.org/api_docs/python/tf/feature_column/input_layer) method. If this pattern is followed then we can instrument this method and harvest the embedding tensors (i.e., right at the step where they are combined and returned by the method). (Note that the instrumentation can be done via [monkey patching](https://en.wikipedia.org/wiki/Monkey_patch) the method, i.e., replace it dynamically at runtime, so that it does not involve modifying the core TF library.)


## Adding a placeholder for analysis:

Our code snippets describe how to apply integrated gradients using a placeholder. Sometimes production models take input from a FIFO queue rather than a simple placeholder. In such cases, the [TF graph editor API](https://www.tensorflow.org/api_guides/python/contrib.graph_editor) can be used to (locally) swap the queue based feed with a simple placeholder feed.

## Why do you need a baseline?

Baseline selection isn't an artifact of the integrated gradients technique. It is fundamental to the process of attribution.  [No discussion of cause is complete without a comparison.](https://plato.stanford.edu/entries/causation-counterfactual/) For instance, consider a scenario where we blame person P for breaking widget W.  This presumes that widget W would not have broken **had person P not done what they did.**  The phrase in bold is the implicit baseline in this case. 

Another example motivating the necessity of a baseline is to characterize situations when the absence of a feature can be informative. When compared against a baseline with the feature present would help identify the feature as important. E.g. in a medical diagnostic model, a diagnosis for H. pylori infection might be ruled out based on the absence of elevated IgA or IgG .


## How do I select a baseline for my model?

Here are some guidelines for selecting baseline depending on the input type:

**Images**: Consider using a black image. It may also make sense to use a noise image as a baseline and even average the attributions over several different noise baselines.

**Text**: Consider using the all-zero baseline (i.e. using all-zero embedding vector). This may seem unintuitive, but nevertheless works. The reason is that the training process causes the input embeddings of unimportant stop words to have a small norm; these are precisely words that tend to "nothing". Some models constrain embeddings to have unit norm. In these cases, a padding term is an adequate baseline. The  other alternative is to use "stop" words.

**Binary and continuous features**: Some times binary and continuous features are binned and then embedded. In these cases, follow the approach for 'text features above. The other alternative is to use several training data inputs as a baseline and to average the attributions over these.

If you have trouble selecting a baseline, reach out to [integrated-gradients@gmail.com](mailto:integrated-gradients@gmail.com).


## Sanity Checking Baselines 

For a classification model, check that the probability distributions across classes has  high entropy. There are two important caveats here. 

*   First, the high entropy check is simply a necessary condition, because one could construct an "[adversarial example](https://blog.openai.com/adversarial-example-research/)"  that satisfies this condition.
*   Second, in some cases, the model may treat one of the classes as a "default" class. In such cases we want the baseline to have the default class label. For instance, consider a toxicity model for text that outputs a sigmoid score. The model may be looking for indicators of toxicity in the text and predicting non-toxic only when there no toxicity indicators. The model treats the non-toxic class as the default as there is no positive evidence for non-toxicity.  In this case a good baseline may be one that has a sigmoid score of 0.0 (meaning non-toxic) as the model may be using the entire range of the sigmoid output (from 0.0 to 1.0) to indicate confidence of toxicity. On the other hand, a text sentiment model may be more symmetric across the positive and negative classes with there being distinct triggers for either class. For such a model a good baseline would be one with a sigmoid score of 0.5 that is balanced between the two classes.

  
## Binary Classification models may need special treatment

Sometimes you have a classification problem with two classes where membership in either class involves the presence of signal. That is, it is not the case that one of the classes is synonymous with absence of signal (no toxicity, no diabetic retinopathy etc.)

Ideally, a good baseline should give equal probability to both classes. In our experience, this may not be true. For instance, we tried Integrated Gradients on a network that classifies images into one of two classes A or B. The black image scored 70% class A.  Then, for images of either prediction class, artifacts of the baseline leak into the attributions, which is undesirable. We would have liked the attributions to be about the input. In this case, we retrained the model to have a third output class C that corresponded to nothing; we supplied black images as true positives for this class. 


## Tuning the number of steps in the gradient approximation

Computing Integrated gradients involves approximating a path integral via a summation. If the underlying function being integrated over is smooth, a sampling the function at a few points suffices. If the underlying function jumps around, we may need many more steps. We have found that anywhere between **20 to 1000 steps** make sense across applications.

The value of the integral should equal the difference between scores at the baseline and the input; this is a theoretical property of Integrated Gradients. If this does not hold within some error bound (say ~5%), then increase the number of steps. 


## Visualizing the Attributions

Often you will not communicate the attribution numbers, but communicate a visualization of the attributions. Take care to choose a good visualization. Bad visualizations can distort the attribution numbers.

For image models, we superimpose the attributions on the image, i.e., a saliency map; this helps view the attributions in comparison to the human-intelligible features of the underlying image. The [visualization library](https://github.com/ankurtaly/Integrated-Gradients/tree/master/VisualizationLibrary) performs a number of operations to make the attributions clearly and faithfully highlight features deemed important by the attributions.  <span style="color:#333333;">See this [notebook](https://github.com/ankurtaly/Integrated-Gradients/blob/master/attributions.ipynb)<span style="color:#333333;"> for details about visualizations<span style="color:#333333;">.</span></span></span>

Further, you can view positive attributions in green, and negative ones in red. Intuitively, the red corresponds to parts of the image, that if moved closer to the baseline value would cause the prediction score to increase. In contrast the green corresponds to parts of the image that if they were moved away from the baseline value, the prediction score would increase.


## How do I visualize the attributions for text models?

For text-based models, attributions can be visualized using colored text to depict the strength of attributions. E.g. below we use a scale from red (very negative attributions) to green (very positive), while gray color is no attribution.:

![](/Visualizations/text_visualization.png)

<!--
<span style="color:#7e817e;">I<span style="color:#212121;"> <span style="color:#7c837c;">am<span style="color:#212121;"> <span style="color:#7e827e;">feeling<span style="color:#212121;"> <span style="color:#08ff08;">super<span style="color:#212121;"> <span style="color:&#639e63;">lucky</span></span></span></span></span></span></span></span></span>

<span style="color:#7a857a;">But<span style="color:#212121;"> <span style="color:#817e7e;">the<span style="color:#212121;"> <span style="color:#926f6f;">results<span style="color:#212121;"> <span style="color:#857a7a;">were<span style="color:#212121;"> <span style="color:#897676;">pretty<span style="color:#212121;"> <span style="color:#f80f0f;">bad</span></span></span></span></span></span></span></span></span></span></span>
-->
The following code can be used to generate this visualization


```
from IPython.display import display, HTML

def visualize_token_attrs(tokens, attrs):
  """
  Visualize attributions for given set of tokens.
  Args:
  - tokens: An array of tokens
  - attrs: An array of attributions, of same size as 'tokens',
    with attrs[i] being the attribution to tokens[i]
  
  Returns:
  - visualization: An IPython.core.display.HTML object showing
    tokens color-coded based on strength of their attribution.
  """
  def get_color(attr):
    if attr > 0:
      r = int(128*attr) + 127
      g = 128 - int(64*attr)
      b = 128 - int(64*attr) 
    else:
      r = 128 + int(64*attr)
      g = 128 + int(64*attr) 
      b = int(-128*attr) + 127
    return r,g,b

  # normalize attributions for visualization.
  bound = max(abs(attrs.max()), abs(attrs.min()))
  attrs = attrs/bound
  html_text = ""
  for i, tok in enumerate(tokens):
    r,g,b = get_color(attrs[i])
    html_text += " <span style='color:rgb(%d,%d,%d)'>%s</span>" % (r, g, b, tok)
  return HTML(html_text)
```


## Common errors

Integrated Gradients has the property that the attributions should sum to the difference in scores of the input and the baseline. Here are common reasons why the attributions may not add up. Check if any of these apply in your case:
*  The sum of the attributions must approximately equal score at the input minus the score at the baseline. If these don't match then it is possible the number of steps used for approximating the integral are too small. Try increasing the number of steps.
*  If the score for the baseline is non-zero, make sure it is accounted for.
*  Implementation of integration via summing the gradients should be sum(gradient * (input-baseline)/steps). It is possible that the scaling factor (input -baseline) is missing. (See for instance the term x<sub>i</sub> - x'<sub>i</sub> in Equation 1 from the Integrated Gradients [paper](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf).)
