import pandas as pd
import requests
import json

df = pd.read_csv('unlabeled_data.csv')
df_sample = df.sample(n = 100)

for k, v in df_sample.iterrows():
    input_json = v.to_dict()
    r = requests.post('http://127.0.0.1:9998/query_model', json = input_json)
    output_json = json.loads(r.text)
    df_sample.loc[df_sample['Ticket number'] == v['Ticket number'], 'prediction'] = output_json['prediction']
    print(input_json)
    print(output_json)
    print()


df_sample.to_csv('prediction_output.csv', index = False)
