"""Riemann + LogReg: recentered tangent-space features, one linear classifier for everyone."""

from sklearn.linear_model import LogisticRegression

from bci.riemann import RiemannFeatures, StreamingDecoder


def fit(train, C=0.01, band=(4, 40), prior_weight=5, seed=0):
    features = RiemannFeatures(band, prior_weight)
    x, _ = features.fit_transform(train)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(x, train.y)
    model = StreamingDecoder(features, lambda x, fp: int(clf.predict(x[None])[0]))
    model.clf = clf
    return model
