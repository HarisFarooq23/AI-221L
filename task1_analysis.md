Model Analysis
Overfitting and Underfitting
Overfitting

Overfitting occurs when a model performs very well on training data but poorly on testing data.

It means the model has learned noise or patterns specific to the training dataset instead of general patterns.

Indicators of overfitting in this experiment:

High training accuracy

Lower testing accuracy

Large magnitude weight values

In logistic regression, overfitting can occur when:

The model uses many features without constraint

There is no regularization applied

Overfitting reduces generalization ability, meaning the model does not perform well on unseen patient data.

Underfitting

Underfitting occurs when the model is too simple to capture the underlying patterns in the data.

It results in:

Low training accuracy

Low testing accuracy

In this experiment, underfitting was demonstrated by using only a subset of features:

age

chol

thalach

Heart disease prediction depends on multiple medical attributes, so limiting the model to only three features reduces predictive performance.

Signs of underfitting observed:

Lower overall accuracy compared to the full-feature model

Lower F1-score

Reduced ROC-AUC

Underfitting indicates high bias in the model.

Effect of Regularization (L2)

L2 regularization adds a penalty term proportional to the square of the weights.

This prevents the weights from becoming excessively large.

It reduces model variance and improves generalization.

Observations from Experiment

For λ = 1:

Slight reduction in weight magnitude

Similar or slightly improved testing performance

Better balance between bias and variance

Improved generalization compared to the non-regularized model

For λ = 10:

Stronger penalty applied to weights

Reduced training accuracy

Possible decrease in testing accuracy

Model becomes simpler

May begin to underfit if λ is too large

Conclusion on Regularization

Small regularization improves model stability and generalization.

Large regularization can oversimplify the model.

Proper selection of λ is important for optimal performance.

Final Conclusions on Model Selection

The simple logistic regression model provides a strong baseline.

The L2 regularized model with λ = 1 provides the best balance between performance and generalization.

The model with λ = 10 may be slightly over-regularized.

The subset-feature model clearly underfits and performs worse.

Final Selected Model

Logistic Regression with L2 Regularization (λ = 1)

Reasons:

Balanced training and testing accuracy

Better generalization performance

Controlled weight magnitudes

Strong and stable ROC-AUC score

Overall Learning Outcome

Feature selection significantly impacts model performance.

Regularization is essential for controlling model complexity.

Both overfitting and underfitting reduce model reliability.

The best model is the one that balances bias and variance while maintaining strong generalization on unseen data.