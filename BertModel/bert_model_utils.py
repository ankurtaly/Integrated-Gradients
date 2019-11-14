# Copyright 2018 The Google AI Language Team Authors
import numpy as np
from IPython.display import HTML


def get_tensor(sess, name):
    """ Returns tensor of given name"""
    return sess.graph.get_tensor_by_name(name)


def get_ig_attributions(sess, input_tensors, embedding_tensor,
                        gradient_tensor, output_tensor, transformed_input_df,
                        baseline_df, tokenizer, max_allowed_error=5,
                        debug=False):
    """ Returns integrated gradients for a single instance input and
        baseline. The

    :param sess: Tensorflow session in which model is loaded

    :param input_tensors: input tensors for the Tensorflow saved model

    :param embedding_tensor: tensor corresponding to embedding layer of model

    :param gradient_tensor: gradient tensor of model output with respect to
        embedding_tensor

    :param output_tensor: tensor corresponding to output of the model

    :param transformed_input_df: Pandas DataFrame returned by the
        transform_input method

    :param baseline_df: Pandas DataFrame returned by the generate_baseline
        method

    :param tokenizer:  Tokenizer used to tokenize the input text. For
            example, the 'BERT-Base, Uncased' tokenizer

    :param max_allowed_error: max error of ig estimation using efficiency axiom

    :param debug: run in debug mode with logging

    :returns a dictionary with a single key 'outputs', mapped
        to a list of two lists, the first one containing the text tokens,
        and the second their corresponding attributions, obtained using the
        Integrated Gradients method.
    """

    num_reps = 1
    integrated_gradients, baseline_prediction, prediction = \
        _compute_ig(sess, input_tensors, embedding_tensor,
                    gradient_tensor, output_tensor, transformed_input_df,
                    baseline_df, num_reps=num_reps)
    error_percentage = \
        _get_ig_error(integrated_gradients, baseline_prediction, prediction,
                      debug=debug)

    while abs(error_percentage) > max_allowed_error:
        num_reps += 5
        if debug:
            print(f'Num reps is {num_reps}, abs error percentage is '
                  f'{error_percentage}')
        integrated_gradients, baseline_prediction, prediction = \
            _compute_ig(sess, input_tensors, embedding_tensor,
                        gradient_tensor, output_tensor, transformed_input_df,
                        baseline_df, num_reps=num_reps)
        error_percentage = \
            _get_ig_error(integrated_gradients, baseline_prediction,
                          prediction, debug=debug)

    integrated_gradients = _project_attributions(tokenizer,
                                                 transformed_input_df,
                                                 integrated_gradients)

    return integrated_gradients


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
            g = int(128*attr) + 127
            b = 128 - int(64*attr)
            r = 128 - int(64*attr)
        else:
            g = 128 + int(64*attr)
            b = 128 + int(64*attr)
            r = int(-128*attr) + 127
        return r,g,b

    # normalize attributions for visualization.
    bound = max(abs(attrs.max()), abs(attrs.min()))
    attrs = attrs/bound
    html_text = ""
    for i, tok in enumerate(tokens):
        r, g, b = get_color(attrs[i])
        html_text += " <span style='color:rgb(%d,%d,%d)'>%s</span>" % \
                     (r, g, b, tok)
    return HTML(html_text)


def transform_input(tokenizer, input_df):
    """
        Transform the provided dataframe into one that complies with the input
        interface of the BERT model.

        Specifically, the BERT model takes four features as input: 'input_ids'
        'input_mask', 'segment_ids', 'label_ids'. This method derives the
        value of these features from the provided text segment in the input
        DataFrame.

        :param tokenizer:  Tokenizer used to tokenize the input text. For
            example, the 'BERT-Base, Uncased' tokenizer

        :param input_df: DataFrame with a single column named 'sentence' that
            contains the text whose prediction is being attributed.

        :returns transformed_input_df: DataFrame having four columns
            'input_ids', 'input_mask', 'segment_ids', 'label_ids' that specify
            the input to the BERT model.

    """
    max_seq_length = 256
    transformed_input_df = input_df.copy(deep=True)
    transformed_input_df['input_ids'] = input_df['sentence'].apply(
        lambda x: ['[CLS]'] + tokenizer.tokenize(x)[:(
                max_seq_length - 2)] + ['[SEP]'])

    transformed_input_df['input_ids'] = transformed_input_df[
        'input_ids'].apply(
        lambda x: tokenizer.convert_tokens_to_ids(x))

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    transformed_input_df['input_mask'] = transformed_input_df[
        'input_ids'].apply(lambda x: [1] * len(x))

    # Zero-pad up to the sequence length.
    transformed_input_df['input_ids'] = transformed_input_df[
        'input_ids'].apply(
        lambda x: x + [0] * (max_seq_length - len(x)))
    transformed_input_df['input_mask'] = transformed_input_df[
        'input_mask'].apply(
        lambda x: x + [0] * (max_seq_length - len(x)))
    transformed_input_df['segment_ids'] = transformed_input_df[
        'input_ids'].apply(lambda x: [0] * max_seq_length)
    transformed_input_df['label_ids'] = 0

    transformed_input_df = transformed_input_df[['input_ids',
                                                 'input_mask',
                                                 'segment_ids',
                                                 'label_ids']]
    return transformed_input_df


def generate_baseline(tokenizer, input_df):
    """
    Generates a baseline for the provided input that is required for
    calculating Integrated Gradients.

    The returned baseline replaces each token in the provided input with a
    padding token ('[PAD]')

    :param tokenizer:  Tokenizer used to tokenize the input text. For
            example, the 'BERT-Base, Uncased' tokenizer

    :param input_df: DataFrame with a single column named 'sentence' that
        contains the text whose prediction is being attributed.

    :returns baseline_df: DataFrame having four columns
        'input_ids', 'input_mask', 'segment_ids', 'label_ids' that specify
        an input to the BERT classification models. These
        values specify a baseline input formed by replacing every token
        in the input sentence with a padding token.
    """
    max_seq_length = 256
    baseline_df = input_df.copy(deep=True)
    len_tokens = len(tokenizer.tokenize(input_df['sentence'][0]))
    baseline_df['input_ids'] = input_df['sentence'].apply(
        lambda x: ['[CLS]'] +
                  ['[PAD]'] * min(len_tokens, (max_seq_length - 2)) +
                  ['[SEP]'])

    baseline_df['input_ids'] = baseline_df['input_ids']. \
        apply(lambda x: tokenizer.convert_tokens_to_ids(x))

    baseline_df['input_mask'] = baseline_df['input_ids']. \
        apply(lambda x: [1] * len(x))

    # Zero-pad up to the sequence length.
    baseline_df['input_ids'] = baseline_df['input_ids']. \
        apply(lambda x: x + [0] * (max_seq_length - len(x)))

    baseline_df['input_mask'] = baseline_df['input_mask']. \
        apply(lambda x: x + [0] * (max_seq_length - len(x)))
    baseline_df['segment_ids'] = baseline_df['input_ids']. \
        apply(lambda x: [0] * max_seq_length)
    baseline_df['label_ids'] = 0

    baseline_df = baseline_df[['input_ids', 'input_mask', 'segment_ids',
                               'label_ids']]
    return baseline_df


def _get_feed_dict(input_tensors, input_df):
    """Returns a dictionary of mapping tensor names to input values for the
    given tensor"""
    feed = {}
    for key, tensor_info in input_tensors.items():
        feed[tensor_info.name] = input_df[key].tolist()
    return feed


def _project_attributions(tokenizer, transformed_input_df, attributions):
    """
    Maps the attributions to the token ids specified in the (transformed)
    input to the corresponding token texts.

    :param tokenizer:  Tokenizer used to tokenize the input text. For
            example, the 'BERT-Base, Uncased' tokenizer

    :param transformed_input_df: DataFrame specifying a BERT model input
        as returned by the transform_input function. It has exactly
        one row as currently only instance explanations are supported.

    :param attributions: dictionary with a single key 'input_ids' mapped
        to a list containing the attributions of the 'input_ids' (i.e.
        token ids)  specified in the transformed_input_df. The order is
        maintained.

    :returns a dictionary with a single key 'outputs', mapped
        to a list of two lists, the first one containing the text tokens,
        and the second their corresponding attributions
    """
    pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    tokens = [tokenizer.convert_ids_to_tokens([int(t)])[0] + " " for
              t in
              transformed_input_df['input_ids'][0] if
              t not in [pad_id, cls_id, sep_id]]

    return {'outputs': [tokens, attributions.astype(
        'float').tolist()[1:len(tokens)+1]]}


def _get_ig_error(integrated_gradients, baseline_prediction, prediction,
                  debug=False):
    sum_attributions = 0
    sum_attributions += np.sum(integrated_gradients)

    delta_prediction = prediction - baseline_prediction

    error_percentage = \
        100 * (delta_prediction - sum_attributions) / delta_prediction
    if debug:
        print(f'prediction is {prediction}')
        print(f'baseline_prediction is {baseline_prediction}')
        print(f'delta_prediction is {delta_prediction}')
        print(f'sum_attributions are {sum_attributions}')
        print(f'Error percentage is {error_percentage}')

    return error_percentage


def _get_scaled_inputs(input_val, baseline_val, batch_size, num_reps):
    list_scaled_embeddings = []
    scaled_embeddings = \
        [baseline_val + (float(i) / (num_reps * batch_size - 1)) *
         (input_val - baseline_val) for i in range(0, num_reps * batch_size)]

    for i in range(num_reps):
        list_scaled_embeddings.append(
            np.array(scaled_embeddings[i * batch_size:i * batch_size +
                                                      batch_size]))

    return list_scaled_embeddings


def _get_unscaled_inputs(input_val, batch_size):
    unscaled_embeddings = [input_val] * batch_size

    return np.array(unscaled_embeddings)


def _calculate_integral(ig):
    # We use np.average here since the width of each
    # step rectangle is 1/number of steps and the height is the gradient,
    # so summing the areas is equivalent to averaging the gradient values.

    ig = (ig[:-1] + ig[1:]) / 2.0  # trapezoidal rule

    integral = np.average(ig, axis=0)

    return integral


def _compute_ig(sess, input_tensors, embedding_tensor,
                gradient_tensor, output_tensor, transformed_input_df,
                baseline_df, num_reps):
    batch_size = 20  # keep small enough to ensure that we do not run out of
    # memory
    num_reps = num_reps

    tensor_values = sess.run(embedding_tensor,
                             _get_feed_dict(input_tensors,
                                            transformed_input_df))

    tensor_baseline_values = sess.run(embedding_tensor,
        _get_feed_dict(input_tensors, baseline_df))

    scaled_embeddings = _get_scaled_inputs(tensor_values[0],
                                           tensor_baseline_values[0],
                                           batch_size, num_reps)
    scaled_input_feed = {}
    for key, tensor_info in input_tensors.items():
        scaled_input_feed[
            get_tensor(sess, tensor_info.name)] = _get_unscaled_inputs(
            transformed_input_df[key][0], batch_size)

    scores = []
    path_gradients = []

    for i in range(num_reps):
        scaled_input_feed[embedding_tensor] = scaled_embeddings[i]
        path_gradients_rep, scores_rep = sess.run(
            [gradient_tensor, output_tensor[:, 1]], scaled_input_feed)
        path_gradients.append(path_gradients_rep[0])
        scores.append(scores_rep)

    baseline_prediction = scores[0][
        0]  # first score is the baseline prediction
    prediction = scores[-1][-1]  # last score is the input prediction

    # integrating the gradients and multiplying with the difference of the
    # baseline and input.
    ig = np.concatenate(path_gradients, axis=0)
    integral = _calculate_integral(ig)
    integrated_gradients = (tensor_values[0] - tensor_baseline_values[
        0]) * integral
    integrated_gradients = np.sum(integrated_gradients, axis=-1)

    return integrated_gradients, baseline_prediction, prediction



