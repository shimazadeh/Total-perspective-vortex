import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score, ShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline, pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets


def pipeline_creation(X, y, transformer1, transformer2, transformer3):
    cv = ShuffleSplit(10, test_size=0.4, random_state=42)

    lda = LinearDiscriminantAnalysis()
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
    rfc = RandomForestClassifier(n_estimators=150, random_state=42)

    scores1 = []
    scores2 = []
    scores3 = []

    pipeline1 = make_pipeline(transformer1, lda)
    pipeline2 = make_pipeline(transformer2, log_reg)
    pipeline3 = make_pipeline(transformer3, rfc)

    scores1 = cross_val_score(pipeline1, X, y, cv=cv, n_jobs=-1)
    scores2 = cross_val_score(pipeline2, X, y, cv=cv, n_jobs=-1)
    scores3 = cross_val_score(pipeline3, X, y, cv=cv, n_jobs=-1)

    print(f"LinearDiscriminantAnalysis: accuracy {scores1.mean()}, std: {scores1.std()}")
    print(f"LogisticRegression        : accuracy {scores2.mean()}, std: {scores2.std()}")
    print(f"RandomForestClassifier    : accuracy {scores3.mean()}, std: {scores3.std()}")

