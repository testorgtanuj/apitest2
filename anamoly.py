# -*- coding: utf-8 -*-
"""

@author: GU20005915
"""
# -*- coding: utf-8 -*-



#import packages 
from flask import Flask, request, jsonify
import pandas as pd 
from scipy import linalg
import scipy as sp 
import numpy  as np
#import sys
import json


# Your API definition
app = Flask(__name__)


@app.route('/mahalanobis', methods=['POST'])
def mahalanobis():
    #Compute the Mahalanobis Distance between each row of x and the data  
    #x    : vector or matrix of data with, say, p columns.
    #data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    #cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    
    
    master = pd.read_csv("./input_file-V1.csv") ## Load CSV file 
    data = master[['head', 'efficiency','flow_ratio_bep','suction_pressure','case_pressure','flow']]
    df_x = master[['head', 'efficiency']].head(1000)
    x_minus_mu = df_x - np.mean(data)
    cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    result = mahal.diagonal()
    thrsld_value = result.mean()+(result.std()*3)
    
    ##head
    json_input = request.json
    print(json_input)
    query = pd.get_dummies(pd.DataFrame(json_input))
    #query=[{"head":470,"efficiency":70,"flow":0.5,"flow_ratio_bep":0.74,"suction_pressure":550428,"case_pressure":4755705}]
    include = ['head','efficiency']
    df_x_v1 = pd.DataFrame(query)  
    df_x_v1 = df_x_v1[include]
    x_minus_mu_v1 = df_x_v1 - np.mean(data)
    left_term_v1 = np.dot(x_minus_mu_v1, inv_covmat)
    mahal = np.dot(left_term_v1, x_minus_mu_v1.T)
    result = pd.DataFrame(mahal.diagonal())
    result.columns =['dist']
    result['pump_head'] = np.where(result.dist > thrsld_value, 'Anomaly', 'Normal')
    #(result)
    
    ##flow
    data = master[['flow', 'flow_ratio_bep']]
    df_x = master[['flow', 'flow_ratio_bep']].head(1000)
    x_minus_mu = df_x - np.mean(data)
    cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    result1 = mahal.diagonal()
    thrsld_value = result1.mean()+(result1.std()*3)
    
  
    #json_input = request.json
    #print(json_input)
    #query = pd.get_dummies(pd.DataFrame(json_input))
    #print("second stage")
    #query=[{"head":470,"efficiency":70,"flow":450}]
    include = ['flow','flow_ratio_bep']
    df_x_v1 = pd.DataFrame(query)  
    df_x_v1 = df_x_v1[include]
    x_minus_mu_v1 = df_x_v1 - np.mean(data)
    left_term_v1 = np.dot(x_minus_mu_v1, inv_covmat)
    mahal = np.dot(left_term_v1, x_minus_mu_v1.T)
    result1 = pd.DataFrame(mahal.diagonal())
    result1.columns =['dist']
    result1['pump_flow'] = np.where(result1.dist > thrsld_value, 'Anomaly', 'Normal')
    

    
    ##pressure
    data = master[['case_pressure', 'suction_pressure']]
    df_x = master[['case_pressure', 'suction_pressure']].head(1000)
    x_minus_mu = df_x - np.mean(data)
    cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    result2 = mahal.diagonal()
    thrsld_value = result2.mean()+(result2.std()*3)
    
  
    #json_input = request.json
    #print(json_input)
    #query = pd.get_dummies(pd.DataFrame(json_input))
    #print("second stage")
    #query=[{"flow":470,"head":70}]
    include = ['case_pressure','suction_pressure']
    df_x_v1 = pd.DataFrame(query)  
    df_x_v1 = df_x_v1[include]
    x_minus_mu_v1 = df_x_v1 - np.mean(data)
    left_term_v1 = np.dot(x_minus_mu_v1, inv_covmat)
    mahal = np.dot(left_term_v1, x_minus_mu_v1.T)
    result2 = pd.DataFrame(mahal.diagonal())
    result2.columns =['dist']
    result2['pump_pressure'] = np.where(result2.dist > thrsld_value, 'Anomaly', 'Normal')
    
    final_result = pd.concat([result['pump_head'], result1['pump_flow'], result2['pump_pressure']], axis=1)
    #jsonfiles = json.loads(final_result.to_json(orient='records'))
    #df_list = result.values.tolist()
    #JSONP_data = jsonpify(df_list)
    return jsonify({'pump_head_status': str(list(final_result.pump_head.values)),'pump_flow_status': str(list(final_result.pump_flow.values)),'pump_pressure_status': str(list(final_result.pump_pressure.values))})

 #if lr:
 #       try:
            #json_ = request.json
            #print(json_)
            #query = pd.get_dummies(pd.DataFrame(json_))
#            query=[{"GR1":124,"GR2":-976,"GR3":-217,"PP1":352,"PP2":448,"PP3":512,"RMS1":67,"RMS2":67,"RMS3":95,"tempValue":38},
#                   {"GR1":124,"GR2":-976,"GR3":-217,"PP1":352,"PP2":448,"PP3":512,"RMS1":67,"RMS2":67,"RMS3":95,"tempValue":38}                  ]
#            query = pd.DataFrame(query)
            #include = ['acousticValue','messageType','tempValue','GR__001','GR__002','GR__003','PP__001','PP__002','PP__003','RMS__001','RMS__002','RMS__003','failureStatus']
            #query = pd.DataFrame(json_)
            #json_ = json_[include]
#            query = query.reindex(columns=model_columns, fill_value=0)
#            prediction = list(lr.predict(query))
#            i=0
#            timeline=[0]
            #for i in prediction:
            # if prediction[i] == 0:
            #  timeline.append('NA')
            # elif prediction[i-1] >= 0.75:
             # timeline.append('<24 hours')
             #else:
             # timeline.append('<24 hours')
            #return jsonify({'prediction': str(prediction),'timeline':timeline})

        #except:

         #   return jsonify({'trace': traceback.format_exc()})
    #else:
     #   print ('Train the model first')
     #  return ('No model here to use')



if __name__ == '__main__':
    #master = pd.read_csv("E:/Physics based models/Mahalanobis Dist/input_file-V1.csv") ## Load CSV file 
    #df_x = master[['head', 'efficiency']].head(1000)
    #df_x['mahala'] = mahalanobis(x=df_x, data=master[['head', 'efficiency']])
    
    #try:
    #   port = int(sys.argv[1]) # This is for a command-line input
    #except:
        #port = 12345 # If you don't provide any port the port will be set to 12345
        app.run(port=5000, debug=True)
    
