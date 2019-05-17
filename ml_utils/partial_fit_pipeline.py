"""Pipeline that supports `partial_fit()`."""

from sklearn.pipeline import Pipeline

class PartialFitPipeline(Pipeline):
    """Subclass of sklearn.pipeline.Pipeline that supports the `partial_fit()` method."""

    def partial_fit(self, X, y):
        """Run `partial_fit()` for online learning estimators when used in a pipeline."""
        for _, step in self.steps[:-1]:
            X = step.fit_transform(X)
        self.steps[-1][1].partial_fit(X, y)
        return self
