import os
import time
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import sys,os
sys.path.append(os.getcwd() + '/model')
import web_methods

layout = go.Layout({"title": "Top expressed genes in iPSC",
                       "yaxis": {"title":"Expression (TPM)"},
                       "showlegend": False})
fig = go.Figure(layout=layout)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H2('StemDB - iPSC prediction test'),
    html.H5('Select or upload your sample:'),
    html.Div(
        dcc.Dropdown(
            id='sample',
            options=[
            {'label': 'iPSC (GSM3576810)', 'value': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3576nnn/GSM3576810/suppl/GSM3576810_1342ed09-a675-4e07-b342-757895f4fa3d.tpm.tsv.gz'},
            {'label': 'iPSC-CM (GSM3576803)', 'value': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3576nnn/GSM3576803/suppl/GSM3576803_00d5b244-97b4-42eb-9a21-370776533f09.tpm.tsv.gz'},
            {'label': 'Fibroblast (GSM2772599)', 'value':'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2772nnn/GSM2772599/suppl/GSM2772599_hFb_MRC5_rep1.genes.results.txt.gz'},
            ],
            value='',
            style={'width': '50%'},
        )),
    html.Div(id='display-value'),
    dcc.Graph(
        id='box',
        figure=fig
    ),    
    html.Div(
            [
        dbc.Spinner(html.Div(id="loading-output")),
    ]),
    html.Br(),
    html.Div(id='model_c', style={'textAlign': 'center'}),
    html.Div(id='model_p', style={'textAlign': 'center'})
])
top=pd.read_csv('data/top20tf.csv',index_col=0)
topi=pd.read_csv('data/top20tf_ens.csv',index_col=0)['Gene stable ID'].values

def get_df(url):
    df=pd.read_csv(url,sep='\t',index_col=0)
    df.index=df.index.str.split('.').str[0]
    if 'tpm' in df.columns:
        tpm='tpm'
    else:
        tpm='TPM'   
    if df.index.name=='target_id' :
        ens=pd.read_csv('data/mart_export.txt',sep='\t',index_col=1)
        ens=ens.rename(columns={'Gene name':'gene','Gene stable ID':'id'})
        ens=ens[~ens.gene.str.startswith('MT')][~ens.gene.str.startswith('RP')][~ens.gene.str.startswith('AC')][~ens.gene.str.startswith('CTD-')]
        df=df.merge(ens[['id']],left_index=True,right_index=True)
        df=df.groupby('id').sum()  
    return tpm, df

@app.callback(
    Output('display-value', 'children'),
    Input('sample', 'value'))
def get_data(sample):
    return 'Sample data: {}'.format(sample)

@app.callback(
    Output('box', 'figure'),
    Output("loading-output", "children"),
    Output('model_c', 'children'),    
    Output('model_p', 'children'),    
    Input('sample', 'value'))
def update_figure(sample):
    fig = go.Figure(layout=layout)
    test_prob=0
    for i in range(20):
        fig.add_trace(
            go.Box(
                y=top.iloc[:,i],
                name=top.columns[i]
            ))
    print(sample)
    if sample!='':
        tpm, df=get_df(sample)
        print(topi)
        fig.add_trace(
            go.Line(
                y=df.loc[topi][tpm],
                x=top.columns
            ))
        fig.update_yaxes(type="log")
        expr_list=df.loc[topi][tpm].values.tolist()
        print(expr_list)
        test_prob = web_methods.get_probability(expr_list, 'data/logistic_v1.joblib')
        print(test_prob)
    return fig, '', "Prediction: "+ str(np.round(test_prob)), "Score: "+ str(test_prob)

#@app.callback(Output('display-value', 'children'),[Input('sample', 'value')])
#def display_value(value):return 'Sample data: {}'.format(value)
#@app.callback(Output("loading-output", "children"), [Input("sample", "value")])
#def load_output(n):time.sleep(1) return ""

if __name__ == '__main__':
    app.run_server(debug=True)
