# tf_1.13_rnn_classifier_bug
Repo to reproduce bug in TF 1.13 regarding RNNClassifier

A tf.contrib.estimator.RNNClassifier is trained on a toy dataset and exported as a SavedModel. We would expect the estimator model and the SavedModel to generate the same probabilities for a sample, but this is not the same. 

For example, the predictions for the estimator on a toy dev dataset are as follow for one instance:

```
{'logits': array([ 0.08922598, -0.09324094, -0.08709991], dtype=float32), 'probabilities': array([0.37431356, 0.31188262, 0.31380382], dtype=float32), 'class_ids': array([0]), 'classes': array([b'positive'], dtype=object)}
{'logits': array([ 0.05508566, -0.09715609, -0.03605224], dtype=float32), 'probabilities': array([0.360793  , 0.30984202, 0.32936496], dtype=float32), 'class_ids': array([0]), 'classes': array([b'positive'], dtype=object)}
{'logits': array([-0.00661559, -0.07290255, -0.03856923], dtype=float32), 'probabilities': array([0.34430358, 0.32222074, 0.33347574], dtype=float32), 'class_ids': array([0]), 'classes': array([b'positive'], dtype=object)}
```

While the predictions for the same dev dataset are as follow for the SavedModel:

```
[array([[0.37773532, 0.18677288, 0.43549177]], dtype=float32), array([[b'positive', b'negative', b'neutral']], dtype=object)]
[array([[0.40707353, 0.31645486, 0.2764716 ]], dtype=float32), array([[b'positive', b'negative', b'neutral']], dtype=object)]
[array([[0.37439883, 0.34236807, 0.28323302]], dtype=float32), array([[b'positive', b'negative', b'neutral']], dtype=object)]
```

When using the same code, except the estimator is changed to a DNNClassifier (and tf.contrib.feature_column.sequence_categorical_column_with_identity is changed to tf.feature_column.categorical_column_with_identity for feature_column), the SavedModel produces the same probabilities as expected. 

```
{'logits': array([ 0.38383368,  0.03943422, -0.08920479], dtype=float32), 'probabilities': array([0.4288621 , 0.30391133, 0.2672266 ], dtype=float32), 'class_ids': array([0]), 'classes': array([b'positive'], dtype=object)}
{'logits': array([ 0.33721772,  0.04164755, -0.09028032], dtype=float32), 'probabilities': array([0.41731942, 0.31053036, 0.27215016], dtype=float32), 'class_ids': array([0]), 'classes': array([b'positive'], dtype=object)}
{'logits': array([ 0.26608875,  0.21454985, -0.05721116], dtype=float32), 'probabilities': array([0.37403825, 0.35524908, 0.2707127 ], dtype=float32), 'class_ids': array([0]), 'classes': array([b'positive'], dtype=object)}

[array([[0.4288621 , 0.30391133, 0.2672266 ]], dtype=float32), array([[b'positive', b'negative', b'neutral']], dtype=object)]
[array([[0.41731942, 0.31053036, 0.27215013]], dtype=float32), array([[b'positive', b'negative', b'neutral']], dtype=object)]
[array([[0.37403825, 0.35524908, 0.2707127 ]], dtype=float32), array([[b'positive', b'negative', b'neutral']], dtype=object)]
```

The code to run the RNNClassifier are located in rnn_classifier.ipynb while the code to run the DNNClassifier are located in dnn_classifier.ipynb.
