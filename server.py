import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import numpy as np
from flask import Flask, request, jsonify
import gc
import sys
import time

app = Flask(__name__)


def get_size(obj, seen=None):
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size


def pad_int(num, l = 4):
    if not pd.isna(num):
        num = int(float(num))
        num_str = str(num)
        while len(num_str) < l:
            num_str  = '0' + num_str
        return num_str


class OHE(OneHotEncoder):
    def __init__(self, n_values=None, categorical_features=None,
                 categories=None, sparse=True, dtype=np.float64,
                 handle_unknown='error', min_perc = .01, col_name = ''):
        super().__init__(n_values=n_values, categorical_features=categorical_features,
                 categories=categories, sparse=sparse, dtype=dtype,
                 handle_unknown=handle_unknown)
        self.min_perc = min_perc
        self.col_name = col_name
        self.valid_values = []
        self.col_names = []
        self.nan_replacement_value = None

    def fit(self, X, y=None):
        input_series = self.process_input(X)
        super().fit( input_series)
        self.col_names = ['{col_base_name}_{value_name}'.format(col_base_name=self.col_name, value_name = i) for i in self.categories_[0]]

    def transform(self, X):
        input_series = self.process_input(X)
        output = super().transform(input_series)
        return self.process_output(output)

    def process_input(self, s):
        if not self.nan_replacement_value:
            self.nan_replacement_value = s.mode()[0]
        s = s.fillna(s.mode())
        s = s.astype(str)

        if not self.valid_values:
            self.valid_values = [i for  i, j in dict(s.value_counts(normalize = True)).items() if j >= self.min_perc]

        prediction_values_to_replace = [i for i in s.unique() if i not in self.valid_values]
        replace_dict = {i: replacement_value for i in prediction_values_to_replace}
        replace_dict.update({i:i for i in self.valid_values})
        s = s.map(replace_dict.get)
        return s.values.reshape(-1, 1)

    def process_output(self, output):
        output_df = pd.DataFrame(data = output.toarray(),
                                columns = self.col_names)
        return output_df


class DataManager():

    def __init__(self, df_train, df_val, problem_setup = 'full_group'):
        self.start_time = time.time()
        self.df_train = df_train
        self.df_val = df_val
        self.problem_setup = problem_setup
        self.model = RandomForestClassifier(n_estimators = 10, max_depth=12, random_state = 1)

        # print('data manager start, data manager size: {0}, {1}'.format(get_size(self), time.time() - self.start_time))
        # self.outlier_model = IsolationForest(random_state = 1, n_estimators = 10)
        self.fit_data_pipeline()

        del self.df_train
        self.df_train = pd.DataFrame()

        gc.collect()
        self.evaluate()


    def fit_data_pipeline(self):
        #location

        self.top_makes = self.df_train['Make'].value_counts()[:25].index.tolist()
        print('self.top_makes, data manager size: {0}, run time: {1}'.format(get_size(self), time.time() - self.start_time))

        # self.df_train.loc[:,'Latitude'] = self.df_train.loc[:,'Latitude'].replace(99999.0, np.nan)
        # self.df_train.loc[:,'Longitude'] = self.df_train.loc[:,'Longitude'].replace(99999.0, np.nan)
        # self.df_train.loc[(self.df_train['Latitude'].notnull()) & (self.df_train['Longitude'].notnull()), 'lat_long_outlier_score'] = self.outlier_model.fit_predict(self.df_train.loc[(self.df_train['Latitude'].notnull()) & (self.df_train['Longitude'].notnull()), ['Latitude', 'Longitude']])
        # print('location trained, data manager size: {0}, run time: {1}'.format(get_size(self), time.time() - self.start_time))

        self.df_train['ticket_dt'] = pd.to_datetime(self.df_train['Issue Date'], errors='coerce')
        self.df_train['ticket_year'] = self.df_train['ticket_dt'].dt.year
        self.df_train['ticket_month'] = self.df_train['ticket_dt'].dt.month
        self.df_train['ticket_dow'] = self.df_train['ticket_dt'].dt.dayofweek
        self.df_train['ticket_hour_of_day'] = self.df_train['Issue time'].apply(lambda x: pad_int(x)).astype(str).str[:2]

        self.df_train.loc[:, 'plate_expiration_diff_dt'] = pd.to_datetime(self.df_train.loc[:,'Plate Expiry Date'].astype(int, errors = 'ignore'), format = '%Y%m', errors='coerce')
        self.df_train.loc[:, 'plate_expiration_diff_dt'] = self.df_train.loc[:,'plate_expiration_diff_dt'] - self.df_train.loc[:,'plate_expiration_diff_dt']
        self.df_train.loc[:, 'plate_expiration_diff_ts'] = self.df_train.loc[:,'plate_expiration_diff_dt'].values.astype(np.int64)
        # self.df_train.loc[:, 'ticket_ts'] = self.df_train.loc[:,'ticket_dt'].values.astype(np.int64)
        # print('plate_expiration_diff calculated, data manager size: {0}'.format(get_size(self)))

        self.rp_state_plate_ohe = OHE(col_name = 'rp_state_plate')
        self.color_ohe = OHE(col_name = 'color')
        self.agency_ohe = OHE(col_name = 'agency')
        self.route_ohe = OHE(col_name = 'route')
        self.violation_code_ohe = OHE(col_name = 'violation_code')
        self.violation_desc_ohe = OHE(col_name = 'violation_desc')
        self.body_style_ohe = OHE(col_name = 'body_style')
        self.ticket_year_ohe = OHE(col_name = 'ticket_year')
        self.ticket_month_ohe = OHE(col_name = 'ticket_month')
        self.ticket_dow_ohe = OHE(col_name = 'ticket_dow')
        self.ticket_hour_of_day_ohe = OHE(col_name = 'ticket_hour_of_day')

        # self.numeric_cols = ['Fine amount', 'lat_long_outlier_score', 'plate_expiration_diff_ts']
        self.numeric_cols = ['Fine amount', 'plate_expiration_diff_ts']
        # print('all OHE calculated, data manager size: {0}, run time: {1}'.format(get_size(self), time.time() - self.start_time))
        self.rp_state_plate_ohe.fit(self.df_train['RP State Plate']),
        self.color_ohe.fit(self.df_train['Color']),
        self.agency_ohe.fit(self.df_train['Agency']),
        self.route_ohe.fit(self.df_train['Route']),
        self.violation_code_ohe.fit(self.df_train['Violation code']),
        self.violation_desc_ohe.fit(self.df_train['Violation Description']),
        self.body_style_ohe.fit(self.df_train['Body Style']),
        self.ticket_year_ohe.fit(self.df_train['ticket_year']),
        self.ticket_month_ohe.fit(self.df_train['ticket_month']),
        self.ticket_dow_ohe.fit(self.df_train['ticket_dow']),
        self.ticket_hour_of_day_ohe.fit(self.df_train['ticket_hour_of_day'])

        train_dfs = [self.rp_state_plate_ohe.transform(self.df_train['RP State Plate']),
                    self.color_ohe.transform(self.df_train['Color']),
                    self.agency_ohe.transform(self.df_train['Agency']),
                    self.route_ohe.transform(self.df_train['Route']),
                    self.violation_code_ohe.transform(self.df_train['Violation code']),
                    self.violation_desc_ohe.transform(self.df_train['Violation Description']),
                    self.body_style_ohe.transform(self.df_train['Body Style']),
                    self.ticket_year_ohe.transform(self.df_train['ticket_year']),
                    self.ticket_month_ohe.transform(self.df_train['ticket_month']),
                    self.ticket_dow_ohe.transform(self.df_train['ticket_dow']),
                    self.ticket_hour_of_day_ohe.transform(self.df_train['ticket_hour_of_day'])]

        numeric_df = self.df_train[self.numeric_cols].reset_index(drop = True)
        train_dfs.append(numeric_df)
        train_data = pd.concat(train_dfs, axis = 1)
        train_labels = self.process_label(self.df_train)

        print('labels and data created, data manager size: {0}, run time: {1}'.format(get_size(self), time.time() - self.start_time))
        #
        # train_data_copy = train_data.copy()
        # train_data_copy['Make'] = self.df_train['Make']
        # train_data_copy.to_csv('label_analysis.csv')

        self.train_nan_fill_choice = train_data.median()
        # self.train_nan_fill_choice['Issue time'] = train_data['Issue time'].mode()

        train_data = train_data.fillna(self.train_nan_fill_choice)#TODO: fix actual nan
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

        print('undersampling_done, data manager size: {0}, run time: {1}'.format(get_size(self), time.time() - self.start_time))

        self.model.fit(train_data, train_labels)
        print('model fit, data manager size: {0}, run time: {1}'.format(get_size(self), time.time() - self.start_time))

        features_importances = dict()
        for i, j in zip(train_data.columns, self.model.feature_importances_):
            features_importances[i] = j

        # feature_impact = []
        # for i in train_data.columns:
        #     slope, intercept, r_value, p_value, std_err = stats.linregress(train_data[i].values, train_labels)
        #     feature_impact.append({'slope':slope,
        #                            'intercept':intercept,
        #                            'r_value':r_value,
        #                            'p_value':p_value,
        #                            'std_err':std_err,
        #                            'columns':i,
        #                            'model_feature_importance':features_importances[i]})
        # analysis_df = pd.DataFrame.from_dict(feature_impact)
        # analysis_df.to_csv('feature_analysis.csv', index = False)


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
        f1_score = 2 * (precision*recall)/(precision + recall)
        print(tn, fp, fn, tp )
        print('Metrics, precision: {precision}, recall: {recall}, f1_score: {f1_score}'.format(precision=precision, recall = recall, f1_score = f1_score))

    def process_label(self, df):
        if self.problem_setup == 'full_group':
            y = df['Make'].isin(self.top_makes).astype(int)
            return y

    def process_features(self, df):
        # df.loc[(df['Latitude'].notnull()) & (df['Longitude'].notnull()), 'lat_long_outlier_score'] = self.outlier_model.predict(df.loc[(df['Latitude'].notnull()) & (df['Longitude'].notnull()), ['Latitude', 'Longitude']])

        df.loc[:, 'ticket_dt'] = pd.to_datetime(df['Issue Date'], errors='coerce')
        df.loc[:, 'plate_expiration_diff_dt'] = pd.to_datetime(df['Plate Expiry Date'].astype(int, errors = 'ignore'), format = '%Y%m', errors='coerce')
        df.loc[:, 'plate_expiration_diff_dt'] = df['plate_expiration_diff_dt'] - df['plate_expiration_diff_dt']
        df.loc[:, 'plate_expiration_diff_ts'] = df['plate_expiration_diff_dt'].values.astype(np.int64)

        df.loc[:, 'ticket_dt'] = pd.to_datetime(df.loc[:,'Issue Date'], errors='coerce')
        df['ticket_year'] = df['ticket_dt'].dt.year
        df['ticket_month'] = df['ticket_dt'].dt.month
        df['ticket_dow'] = df['ticket_dt'].dt.dayofweek
        df['ticket_hour_of_day'] = df['Issue time'].apply(lambda x: pad_int(x)).astype(str).str[:2]

        # df.loc[:, 'ticket_ts'] = df['ticket_dt'].values.astype(np.int64)

        dfs = [self.rp_state_plate_ohe.transform(df['RP State Plate']),
                    self.color_ohe.transform(df['Color']),
                    self.agency_ohe.transform(df['Agency']),
                    self.route_ohe.transform(df['Route']),
                    self.violation_code_ohe.transform(df['Violation code']),
                    self.violation_desc_ohe.transform(df['Violation Description']),
                    self.body_style_ohe.transform(df['Body Style']),
                    self.ticket_year_ohe.transform(df['ticket_year']),
                    self.ticket_month_ohe.transform(df['ticket_month']),
                    self.ticket_dow_ohe.transform(df['ticket_dow']),
                    self.ticket_hour_of_day_ohe.transform(df['ticket_hour_of_day'])]


        numeric_df = df[self.numeric_cols].reset_index(drop = True)
        dfs.append(numeric_df)
        data = pd.concat(dfs, axis = 1)
        data = data.fillna(self.train_nan_fill_choice)
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
                            issue_time = None,
                            rp_state_plate=None):

        new_df = pd.DataFrame(data = [[lat, long, issue_date, plate_expiry_date, color, agency,
                                      route, violation_code, violation_description, body_styles,
                                      fine_amount, issue_time, rp_state_plate]],
                              columns = ['Latitude', 'Longitude', 'Issue Date', 'Plate Expiry Date', 'Color',
                                         'Agency', 'Route', 'Violation code', 'Violation Description',
                                         'Body Style', 'Fine amount', 'Issue time', 'RP State Plate'],
                              index = [0])
        new_data = self.process_features(new_df)
        return self.model.predict_proba(new_data)



@app.route('/query_model', methods = ['POST'])
def query_model():
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
    rp_state_plate = input_json.get('RP State Plate')

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
                            issue_time = issue_time,
                            rp_state_plate=rp_state_plate)
    return jsonify({'prediction': prediction[0][1]})


if __name__ == '__main__':
    replacement_value = 'dummy_replacement_value'
    path = '/home/td/Documents'
    url = 'https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv'
    # df = pd.read_csv('{path}/tickets.csv'.format(path=path), low_memory=False, nrows = 100000)
    df = pd.read_csv(url, low_memory=False)
    df_unlabeled = df[df['Make'].isna()]
    df_unlabeled.to_csv('unlabeled_data.csv', index = False)
    df_labeled = df.dropna(subset = ['Make'])
    df_labeled = df_labeled.sample(frac = 1, random_state = 1)
    del df, df_unlabeled
    gc.collect()
    df_analysis = df_labeled[:int(df_labeled.shape[0] * .9)]
    df_holdout = df_labeled[int(df_labeled.shape[0] * .9):]
    del df_labeled
    gc.collect()

    # print(df_labeled.shape, df_analysis.shape, df_holdout.shape)
    dm = DataManager(df_analysis, df_holdout)
    app.run(host= '127.0.0.1', port = 9992, debug=False)
