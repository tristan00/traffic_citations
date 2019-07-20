import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score



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
        self.model = RandomForestClassifier()
        self.fit_data_pipeline()
        self.evaluate()

    def fit_data_pipeline(self):
        self.rp_state_plate_ohe = OHE(self.df_train['RP State Plate'].copy(), col_name = 'rp_state_plate')
        self.color_ohe = OHE(self.df_train['Color'].copy(), col_name = 'color')
        self.agency_ohe = OHE(self.df_train['Agency'].copy(), col_name = 'agency')
        self.route_ohe = OHE(self.df_train['Route'].copy(), col_name = 'route')
        self.violation_code_ohe = OHE(self.df_train['Violation code'].copy(), col_name = 'violation_code')
        self.violation_desc_ohe = OHE(self.df_train['Violation Description'].copy(), col_name = 'violation_desc')
        self.numeric_cols = ['Fine amount']

        #TODO: add location features
        train_dfs = [self.rp_state_plate_ohe.transform(self.df_train['RP State Plate'].copy()),
                    self.color_ohe.transform(self.df_train['Color'].copy()),
                    self.agency_ohe.transform(self.df_train['Agency'].copy()),
                    self.route_ohe.transform(self.df_train['Route'].copy()),
                    self.violation_code_ohe.transform(self.df_train['Violation code'].copy()),
                    self.violation_desc_ohe.transform(self.df_train['Violation Description'].copy())]

        numeric_df = self.df_train[self.numeric_cols].reset_index(drop = True)
        train_dfs.append(numeric_df)
        train_data = pd.concat(train_dfs, axis = 1)
        train_labels = self.process_label(self.df_train)

        train_data = train_data.fillna(train_data.median())#TODO: fix actual nan
        print([i.shape for i in train_dfs], train_data.shape)

        self.model.fit(train_data, train_labels)

    def evaluate(self):
        val_data = self.process_features(self.df_val)
        val_label = self.process_label(self.df_val)
        preds = self.model.predict(val_data)

        print(val_data.shape, val_label.shape, preds.shape)
        print(f1_score(preds, val_label))


    def process_label(self, df):
        if self.problem_setup == 'full_group':
            top_makes = df['Make'].value_counts()[:25].index.tolist()
            y = df['Make'].isin(top_makes).astype(int)
            return y

    def process_features(self, df):
        dfs = [self.rp_state_plate_ohe.transform(df['RP State Plate'].copy()),
                    self.color_ohe.transform(df['Color'].copy()),
                    self.agency_ohe.transform(df['Agency'].copy()),
                    self.route_ohe.transform(df['Route'].copy()),
                    self.violation_code_ohe.transform(df['Violation code'].copy()),
                    self.violation_desc_ohe.transform(df['Violation Description'].copy())]

        numeric_df = df[self.numeric_cols].reset_index(drop = True)
        dfs.append(numeric_df)
        data = pd.concat(dfs, axis = 1)
        data = data.fillna(data.median())#TODO: fix actual nan
        return data


replacement_value = 'dummy_repalcement_value'
path = '/home/td/Documents'

df = pd.read_csv('{path}/tickets.csv'.format(path=path))
df_labeled = df.dropna(subset = ['Make'])
df_labeled = df_labeled.sample(frac = 1, random_state = 1)
df_analysis = df_labeled[:int(df_labeled.shape[0] * .75)]
df_holdout = df_labeled[int(df_labeled.shape[0] * .75):]

print(df_analysis.shape, df_holdout.shape)
dm = DataManager(df_analysis, df_holdout)

