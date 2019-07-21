import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)


class OHE():
    def __init__(self, training_series, min_perc = .01, col_name = '', nan_treatment = 'mode'):
        if nan_treatment == 'mode':
            training_series = training_series.fillna(training_series.mode())
        training_series = training_series.astype(str)

        self.sk_ohe = OneHotEncoder(handle_unknown = 'ignore')
        self.valid_values = [i for  i, j in dict(training_series.value_counts(normalize = True)).items() if j >= min_perc]
        training_values_to_replace = [i for i in training_series.unique() if i not in self.valid_values]
        training_series = training_series.replace(training_values_to_replace, [replacement_value for _ in training_values_to_replace])
        training_np = training_series.values.reshape(-1, 1)
        self.sk_ohe.fit(training_np)
        self.col_names = ['{col_base_name}_{value_name}'.format(col_base_name=col_name, value_name = i) for i in self.sk_ohe.categories_[0]]

    def transform(self, prediction_series):
        prediction_values_to_replace = [i for i in prediction_series.unique() if i not in self.valid_values]
        prediction_series = prediction_series.replace(prediction_values_to_replace, [replacement_value for _ in prediction_values_to_replace])
        prediction_np = prediction_series.values.reshape(-1, 1)
        output = self.sk_ohe.transform(prediction_np).toarray()
        output_df = pd.DataFrame(data = output,
                                columns = self.col_names)
        return output_df


class DataManager():

    def __init__(self, df_train, df_val, problem_setup = 'full_group'):
        self.df_train = df_train
        self.df_val = df_val
        self.problem_setup = problem_setup
        self.model = RandomForestClassifier(n_estimators = 100)
        self.outlier_model = IsolationForest()
        self.fit_data_pipeline()
        self.evaluate()


    def fit_data_pipeline(self):
        #location
        self.top_makes = self.df_train['Make'].value_counts()[:25].index.tolist()
        self.df_train.loc[:,'Latitude'] = self.df_train.loc[:,'Latitude'].replace(99999.0, np.nan)
        self.df_train.loc[:,'Longitude'] = self.df_train.loc[:,'Longitude'].replace(99999.0, np.nan)
        self.df_train.loc[(self.df_train['Latitude'].notnull()) & (self.df_train['Longitude'].notnull()), 'lat_long_outlier_score'] = self.outlier_model.fit_predict(self.df_train.loc[(self.df_train['Latitude'].notnull()) & (self.df_train['Longitude'].notnull()), ['Latitude', 'Longitude']])

        self.df_train.loc[:, 'ticket_dt'] = pd.to_datetime(self.df_train.loc[:,'Issue Date'], errors='coerce')
        self.df_train.loc[:, 'plate_expiration_diff_dt'] = pd.to_datetime(self.df_train.loc[:,'Plate Expiry Date'].astype(int, errors = 'ignore'), format = '%Y%m', errors='coerce')
        self.df_train.loc[:, 'plate_expiration_diff_dt'] = self.df_train.loc[:,'plate_expiration_diff_dt'] - self.df_train.loc[:,'plate_expiration_diff_dt']
        self.df_train.loc[:, 'plate_expiration_diff_ts'] = self.df_train.loc[:,'plate_expiration_diff_dt'].values.astype(np.int64)
        # self.df_train.loc[:, 'ticket_ts'] = self.df_train.loc[:,'ticket_dt'].values.astype(np.int64)

        self.rp_state_plate_ohe = OHE(self.df_train['RP State Plate'].copy(), col_name = 'rp_state_plate')
        self.color_ohe = OHE(self.df_train['Color'].copy(), col_name = 'color')
        self.agency_ohe = OHE(self.df_train['Agency'].copy(), col_name = 'agency')
        self.route_ohe = OHE(self.df_train['Route'].copy(), col_name = 'route')
        self.violation_code_ohe = OHE(self.df_train['Violation code'].copy(), col_name = 'violation_code')
        self.violation_desc_ohe = OHE(self.df_train['Violation Description'].copy(), col_name = 'violation_desc')
        self.body_style_ohe = OHE(self.df_train['Body Style'].copy(), col_name = 'body_style')

        self.numeric_cols = ['Fine amount', 'lat_long_outlier_score', 'plate_expiration_diff_ts', 'Issue time']

        train_dfs = [self.rp_state_plate_ohe.transform(self.df_train['RP State Plate'].copy()),
                    self.color_ohe.transform(self.df_train['Color'].copy()),
                    self.agency_ohe.transform(self.df_train['Agency'].copy()),
                    self.route_ohe.transform(self.df_train['Route'].copy()),
                    self.violation_code_ohe.transform(self.df_train['Violation code'].copy()),
                    self.violation_desc_ohe.transform(self.df_train['Violation Description'].copy()),
                    self.body_style_ohe.transform(self.df_train['Body Style'].copy())]

        numeric_df = self.df_train[self.numeric_cols].reset_index(drop = True)
        train_dfs.append(numeric_df)
        train_data = pd.concat(train_dfs, axis = 1)
        train_labels = self.process_label(self.df_train)

        train_data_copy = train_data.copy()
        train_data_copy['Make'] = self.df_train['Make']
        train_data_copy.to_csv('label_analysis.csv')

        self.train_nan_fillers = train_data.median()
        train_data = train_data.fillna(self.train_nan_fillers)#TODO: fix actual nan
        train_data['target'] = train_labels
        train_data_positive = train_data[train_data['target'] == 1]
        train_data_negative = train_data[train_data['target'] == 0]

        if train_data_positive.shape[0] > train_data_negative.shape[0]:
            train_data_positive = train_data_positive.sample(n = train_data_negative.shape[0], random_state = 1)
        else:
            train_data_negative = train_data_negative.sample(n = train_data_positive.shape[0], random_state = 1)

        train_data = pd.concat([train_data_positive, train_data_negative])
        train_labels = train_data['target']
        train_data = train_data.drop('target', axis = 1)
        self.model.fit(train_data, train_labels)

        features_importances = dict()
        for i, j in zip(train_data.columns, self.model.feature_importances_):
            features_importances[i] = j

        feature_impact = []
        for i in train_data.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(train_data[i].values, train_labels)
            feature_impact.append({'slope':slope,
                                   'intercept':intercept,
                                   'r_value':r_value,
                                   'p_value':p_value,
                                   'std_err':std_err,
                                   'columns':i,
                                   'model_feature_importance':features_importances[i]})
        analysis_df = pd.DataFrame.from_dict(feature_impact)
        analysis_df.to_csv('feature_analysis.csv', index = False)

    def evaluate(self):
        val_data = self.process_features(self.df_val)
        val_label = self.process_label(self.df_val)
        preds = self.model.predict(val_data)

        print(val_data.shape, val_label.shape, preds.shape)
        print(classification_report(preds, val_label))
        print(confusion_matrix(preds, val_label))
        tn, fp, fn, tp = confusion_matrix(preds, val_label).ravel()
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        f1_score = 2* (precision*recall)/(precision + recall)
        print(tn, fp, fn, tp )
        print('Positive metrics, precision: {precision}, recall: {recall}, f1_score: {f1_score}'.format(precision=precision, recall = recall, f1_score = f1_score))

    def process_label(self, df):
        if self.problem_setup == 'full_group':
            y = df['Make'].isin(self.top_makes).astype(int)
            return y

    def process_features(self, df):
        df.loc[(df['Latitude'].notnull()) & (df['Longitude'].notnull()), 'lat_long_outlier_score'] = self.outlier_model.predict(df.loc[(df['Latitude'].notnull()) & (df['Longitude'].notnull()), ['Latitude', 'Longitude']])

        df.loc[:, 'ticket_dt'] = pd.to_datetime(df['Issue Date'], errors='coerce')
        df.loc[:, 'plate_expiration_diff_dt'] = pd.to_datetime(df['Plate Expiry Date'].astype(int, errors = 'ignore'), format = '%Y%m', errors='coerce')
        df.loc[:, 'plate_expiration_diff_dt'] = df['plate_expiration_diff_dt'] - df['plate_expiration_diff_dt']
        df.loc[:, 'plate_expiration_diff_ts'] = df['plate_expiration_diff_dt'].values.astype(np.int64)
        # df.loc[:, 'ticket_ts'] = df['ticket_dt'].values.astype(np.int64)

        dfs = [self.rp_state_plate_ohe.transform(df['RP State Plate'].copy()),
                    self.color_ohe.transform(df['Color'].copy()),
                    self.agency_ohe.transform(df['Agency'].copy()),
                    self.route_ohe.transform(df['Route'].copy()),
                    self.violation_code_ohe.transform(df['Violation code'].copy()),
                    self.violation_desc_ohe.transform(df['Violation Description'].copy()),
                    self.body_style_ohe.transform(df['Body Style'].copy())]


        numeric_df = df[self.numeric_cols].reset_index(drop = True)
        dfs.append(numeric_df)
        data = pd.concat(dfs, axis = 1)
        data = data.fillna(self.train_nan_fillers)
        return data

    def predict_record(self,lat = None,
                            long =  None,
                            issue_date =  None,
                            plate_expiry_date =  None,
                            color =  None,
                            agency =  None,
                            route =  None,
                            violation_code =  None,
                            violation_description =  None,
                            body_styles =  None,
                            fine_amount =  None,
                            issue_time = None):

        new_df = pd.DataFrame(data = [lat, long, issue_date, plate_expiry_date, color, agency,
                                      route, violation_code, violation_description, body_styles,
                                      fine_amount, issue_time],
                              columns = ['Latitude', 'Longitude', 'Issue Date', 'Plate Expiry Date', 'Color',
                                         'Agency', 'Route', 'Violation code', 'Violation Description',
                                         'Body Style', 'Fine amount', 'Issue time'])
        new_data = self.process_features(new_df)
        return self.model.predict_proba(new_data)



@app.route('/query_model', methods = ['POST'])
def query_model():
    '''

    :return:
    '''
    input_json = request.json
    lat = input_json.get('Latitude')
    long = input_json.get('Longitude')
    issue_date = input_json.get('Issue Date')
    plate_expiry_date = input_json.get('Plate Expiry Date')
    color = input_json.get('Color')
    agency = input_json.get('Agency')
    route = input_json.get('Route')
    violation_code = input_json.get('Violation code')
    violation_description = input_json.get('Violation Description')
    body_styles = input_json.get('Body Style')
    fine_amount = input_json.get('Fine amount')
    issue_time = input_json.get('Issue time')
    prediction = dm.predict_record(lat = lat,
                            long =  long,
                            issue_date =  issue_date,
                            plate_expiry_date =  plate_expiry_date,
                            color =  color,
                            agency =  agency,
                            route =  route,
                            violation_code =  violation_code,
                            violation_description =  violation_description,
                            body_styles =  body_styles,
                            fine_amount =  fine_amount,
                            issue_time = issue_time)

    return jsonify({'result': prediction})


if __name__ == '__main__':
    replacement_value = 'dummy_replacement_value'
    path = '/home/td/Documents'
    df = pd.read_csv('{path}/tickets.csv'.format(path=path), low_memory=False)
    df_labeled = df.dropna(subset = ['Make'])
    df_labeled = df_labeled.sample(frac = 1, random_state = 1)
    df_analysis = df_labeled[:int(df_labeled.shape[0] * .9)]
    df_holdout = df_labeled[int(df_labeled.shape[0] * .9):]
    print(df_labeled.shape, df_analysis.shape, df_holdout.shape)
    dm = DataManager(df_analysis, df_holdout)
    app.run(host= '0.0.0.0',debug=True)
