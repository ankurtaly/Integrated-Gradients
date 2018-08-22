import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from cStringIO import StringIO
from IPython.display import display
from IPython.display import Image
from scipy import ndimage


def ConvertToGrayscale(attributions):
  return np.average(attributions, axis=2)


def Polarity(attributions, polarity):
  if polarity == 'positive':
    return np.clip(attributions, 0, 1)
  elif polarity == 'negative':
    return np.clip(attributions, -1, 0)
  else:
    raise ValueError('Unrecognized polarity option.')


def ComputeThresholdByTopPercentage(attributions,
                                    percentage=60,
                                    plot_distribution=True):
  """Compute the threshold value that maps to the top percentage of values.

  This function takes the cumulative sum of attributions and computes the set
  of top attributions that contribute to the given percentage of the total sum.
  The lowest value of this given set is returned.

  Args:
    attributions: (numpy.array) The provided attributions.
    percentage: (float) Specified percentage by which to threshold.
    plot_distribution: (bool) If true, plots the distribution of attributions
      and indicates the threshold point by a vertical line.

  Returns:
    (float) The threshold value.

  Raises:
    ValueError: if percentage is not in [0, 100].
  """
  if percentage < 0 or percentage > 100:
    raise ValueError('percentage must be in [0, 100]')
  
  # For percentage equal to 100, this should in theory return the lowest
  # value as the threshold. However, due to precision errors in numpy's cumsum,
  # the last value won't sum to 100%. Thus, in this special case, we force the
  # threshold to equal the min value.
  if percentage == 100:
    return np.min(attributions)

  flat_attributions = attributions.flatten()
  attribution_sum = np.sum(flat_attributions)
  
  # Sort the attributions from largest to smallest.
  sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]

  # Compute a normalized cumulative sum, so that each attribution is mapped to
  # the percentage of the total sum that it and all values above it contribute.
  cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
  threshold_idx = np.where(cum_sum >= percentage)[0][0]
  threshold = sorted_attributions[threshold_idx]

  if plot_distribution:
    # Generate a plot of sorted intgrad scores.
    values_to_plot = np.where(cum_sum >= 95)[0][0]
    values_to_plot = max(values_to_plot, threshold_idx)
    plt.plot(np.arange(values_to_plot), sorted_attributions[:values_to_plot])
    plt.axvline(x=threshold_idx)
    plt.show()

  return threshold
    

def LinearTransform(attributions,
                    clip_above_percentile=99.9,
                    clip_below_percentile=70.0,
                    low=0.2,
                    plot_distribution=False):
  """Transform the attributions by a linear function.

  Transform the attributions so that the specified percentage of top attribution
  values are mapped to a linear space between `low` and 1.0.

  Args:
    attributions: (numpy.array) The provided attributions.
    percentage: (float) The percentage of top attribution values.
    low: (float) The low end of the linear space.

  Returns:
    (numpy.array) The linearly transformed attributions.

  Raises:
    ValueError: if percentage is not in [0, 100].
  """
  if clip_above_percentile < 0 or clip_above_percentile > 100:
    raise ValueError('clip_above_percentile must be in [0, 100]')
    
  if clip_below_percentile < 0 or clip_below_percentile > 100:
    raise ValueError('clip_below_percentile must be in [0, 100]')

  if low < 0 or low > 1:
    raise ValueError('low must be in [0, 1]')

  m = ComputeThresholdByTopPercentage(attributions,
                                      percentage=100-clip_above_percentile,
                                      plot_distribution=plot_distribution)
  e = ComputeThresholdByTopPercentage(attributions,
                                      percentage=100-clip_below_percentile,
                                      plot_distribution=plot_distribution)

  # Transform the attributions by a linear function f(x) = a*x + b such that
  # f(m) = 1.0 and f(e) = low. Derivation:
  #   a*m + b = 1, a*e + b = low  ==>  a = (1 - low) / (m - e)
  #                               ==>  b = low - (1 - low) * e / (m - e)
  #                               ==>  f(x) = (1 - low) (x - e) / (m - e) + low
  transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low

  # Recover the original sign of the attributions.
  transformed *= np.sign(attributions)

  # Map values below low to 0.
  transformed *= (transformed >= low)
  
  # Clip values above and below.
  transformed = np.clip(transformed, 0.0, 1.0)
  return transformed


def Binarize(attributions, threshold=0.001):
  return attributions > threshold


def MorphologicalCleanup(attributions, structure=np.ones((4,4))):
  closed = ndimage.grey_closing(attributions, structure=structure)
  opened = ndimage.grey_opening(closed, structure=structure)
  
  return opened


def Outlines(attributions, percentage=90,
             connected_component_structure=np.ones((3,3)),
             plot_distribution=True):
  # Binarize the attributions mask if not already.
  attributions = Binarize(attributions)

  attributions = ndimage.binary_fill_holes(attributions)
  
  # Compute connected components of the transformed mask.
  connected_components, num_cc = ndimage.measurements.label(
      attributions, structure=connected_component_structure)

  # Go through each connected component and sum up the attributions of that
  # component.
  overall_sum = np.sum(attributions[connected_components > 0])
  component_sums = []
  for cc_idx in range(1, num_cc + 1):
    cc_mask = connected_components == cc_idx
    component_sum = np.sum(attributions[cc_mask])
    component_sums.append((component_sum, cc_mask))

  # Compute the percentage of top components to keep.
  sorted_sums_and_masks = sorted(
      component_sums, key=lambda x: x[0], reverse=True)
  sorted_sums = zip(*sorted_sums_and_masks)[0]
  cumulative_sorted_sums = np.cumsum(sorted_sums)
  cutoff_threshold = percentage * overall_sum / 100
  cutoff_idx = np.where(cumulative_sorted_sums >= cutoff_threshold)[0][0]

  if cutoff_idx > 2:
    cutoff_idx = 2
  
  # Turn on the kept components.
  border_mask = np.zeros_like(attributions)
  for i in range(cutoff_idx + 1):
    border_mask[sorted_sums_and_masks[i][1]] = 1

  if plot_distribution:
    plt.plot(np.arange(len(sorted_sums)), sorted_sums)
    plt.axvline(x=cutoff_idx)
    plt.show()

  # Hollow out the mask so that only the border is showing.
  eroded_mask = ndimage.binary_erosion(border_mask, iterations=1)
  border_mask[eroded_mask] = 0
  
  return border_mask


def Overlay(attributions, image):
  return np.clip(0.7 * image + 0.5 * attributions, 0, 255)


def pil_image(x):
  """Returns a PIL image created from the provided RGB array.

  Args:
    x: (numpy.array) RGB array of shape [height, width, 3] consisting of values
      in range 0-255.

  Returns:
    The PIL image.
  """
  x = np.uint8(x)
  return PIL.Image.fromarray(x)


def show_pil_image(pil_img):
  """Display the provided PIL image.

  Args:
    pil_img: (PIL.Image) The provided PIL image.
  """
  f = StringIO()
  pil_img.save(f, 'png')
  display(Image(data=f.getvalue()))


G = [0, 255, 0]
R = [255, 0, 0]
def Visualize(attributions,
              image,
              positive_channel=G,
              negative_channel=R,
              polarity='positive',
              clip_above_percentile=99.9,
              clip_below_percentile=0,
              morphological_cleanup=False,
              structure=np.ones((3,3)),
              outlines=False,
              outlines_component_percentage=90,
              overlay=True,
              plot_distribution=False):
  
  if polarity == 'both':
    pos_attributions = Visualize(
        attributions, image, positive_channel=positive_channel,
        negative_channel=negative_channel, polarity='positive',
        clip_above_percentile=clip_above_percentile, clip_below_percentile=clip_below_percentile,
        morphological_cleanup=morphological_cleanup, outlines=outlines,
        outlines_component_percentage=outlines_component_percentage,
        overlay=False,
        plot_distribution=plot_distribution)
    
    neg_attributions = Visualize(
        attributions, image, positive_channel=positive_channel,
        negative_channel=negative_channel, polarity='negative',
        clip_above_percentile=clip_above_percentile, clip_below_percentile=clip_below_percentile,
        morphological_cleanup=morphological_cleanup, outlines=outlines,
        outlines_component_percentage=outlines_component_percentage,
        overlay=False,
        plot_distribution=plot_distribution)
    
    attributions = pos_attributions + neg_attributions
    
    if overlay:
      attributions = Overlay(attributions, image)
    
    return attributions
  elif polarity == 'positive':
    attributions = Polarity(attributions, polarity=polarity)
    channel = positive_channel
  elif polarity == 'negative':
    attributions = Polarity(attributions, polarity=polarity)
    attributions = np.abs(attributions)
    channel = negative_channel

  attributions = ConvertToGrayscale(attributions)
  
  attributions = LinearTransform(attributions,
                                 clip_above_percentile, clip_below_percentile,
                                 0.0,
                                 plot_distribution=plot_distribution)
  
  if morphological_cleanup:
    attributions = MorphologicalCleanup(attributions, structure=structure)
  if outlines:
    attributions = Outlines(attributions,
                            percentage=outlines_component_percentage,
                            plot_distribution=plot_distribution)
  
  # Convert to RGB space
  attributions = np.expand_dims(attributions, 2) * channel
  
  if overlay:
    attributions = Overlay(attributions, image)

  return attributions
  