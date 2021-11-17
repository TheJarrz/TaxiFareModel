import pandas as pd
import mlflow
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from memoized_property import memoized_property
from mlflow.tracking import MlflowClient

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data




class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"

    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.experiment_name = "[FR] [MAR] [antoine] " + kwargs.get("experiment_name", "test")
        self.X = X
        self.y = y

    def set_pipeline(self, **kwargs):
        """defines the pipeline as a class attribute"""
        estimator = kwargs.get("estimator", "linear_regression")
        if estimator == "linear_regression":
            model = LinearRegression()
        elif estimator == "lasso":
            model = Lasso()
        elif estimator == "elastic_net":
            model = ElasticNet()
        elif estimator == "random_forsest":
            model = RandomForestRegressor()
        elif estimator == "gradient_boost":
            model = GradientBoostingRegressor()
        else:
            assert "Unknown model " + estimator
        model.set_params(**kwargs)

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([('distance',
                                           dist_pipe,
                                           ["pickup_latitude",
                                            "pickup_longitude",
                                            'dropoff_latitude',
                                            'dropoff_longitude']),
                                          ('time',
                                           time_pipe,
                                           ['pickup_datetime'])],
                                         remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                         ('model', model)])
        self.pipeline = pipe
        return self

    def run(self, **kwargs):
        """set and train the pipeline"""
        self.set_pipeline(**kwargs)
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    params = dict(
        nrows=10_000,  # number of samples
        local=True,  # get data from AWS
        optimize=True,
        estimator="gradient_boost",
        mlflow=True,  # set to True to log params to mlflow
        experiment_name="test",
        pipeline_memory=None,
        distance_type="haversine",
        feateng=[
            "distance_to_center", "direction", "distance", "time_features",
            "geohash"
        ])
    df = get_data(**params)
    df = clean_data(df)
    X = df.drop(columns=["fare_amount"])
    y = df.fare_amount

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2)


    best_metric = 0
    best_trainer = None
    trainer = Trainer(X_train, y_train, experiment_name="TaxiFare regression")
    trainer.run(**params)
    rmse = trainer.evaluate(X_test, y_test)
    trainer.mlflow_log_param("model", params["estimator"])
    trainer.mlflow_log_metric("rmse", rmse)
    if not best_trainer:
        best_trainer = trainer
        best_metric = rmse
    if best_metric < rmse:
        best_trainer = trainer

    best_trainer.save_model()

    print(rmse)
