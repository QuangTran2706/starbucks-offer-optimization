# import pandas
from flask import Flask, redirect, request, jsonify
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from helper.helper import *
import pickle

server = Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

X_train, X_test, y_train, y_test = getData(df)
with open("model/lgbmClf_model.pkl", 'rb') as file:
    best_model = pickle.load(file)

def create_app():
    corr_fig = get_corr_fig(df)
    # scatter_fig = get_scatter_figure(df)
    roc_curves_fig = get_roc_figure()
    model_score_fig = get_model_score_figure()
    column_list = [i for i in list(df) if i not in ('customer_id', 'offer id')]
    app.layout = html.Div(
        children=[
            html.Div(
            className='row',
            # Define the row element
            children=[
                # Define the left element
                html.Div(className='four columns div-user-controls',
                children = [
                    html.H1('COSC2789 -Practical Data Science Assignment 3: Group Project'),
                    html.H1('STARBUCKS OFFER PERSONALIZATION'),
                    html.Div(className='div-plot-1-control',
                        children=[html.P('''1. Correlation Matrix Graph.''')]),
                    html.Div(className='div-plot-2-control',
                        children=[html.P('''2. Scatter Graph.'''),
                        html.Div(className='div-for-dropdown',
                            children=
                            [
                                dcc.Dropdown(id='selector_x',
                                options=[{"label": i, "value": i} for i in column_list],
                                multi=False,
                                placeholder="Select x column",)
                            ]),      
                        html.Div(className='div-for-dropdown',
                            children=
                            [
                                dcc.Dropdown(id='selector_y',
                                options=[{"label": i, "value": i} for i in column_list],
                                multi=False,
                                placeholder="Select y column",)
                            ]),                   
                        ]),
                    html.Div(className='div-plot-2-control',
                        children=[html.P('''3. ROC Curves Graph.''')]),
                    html.Div(className='div-plot-2-control',
                        children=[html.P('''4. Model Score Comparasion Graph.''')]),
                ]),
                # Define the right element
                html.Div(className='eight columns div-for-charts bg-grey',
                children = [
                    dcc.Graph(id='heatmap_graph', figure=corr_fig),
                    dcc.Graph(id='scatter_graph'),
                    dcc.Graph(id="roc_train_graph", figure=roc_curves_fig),
                    dcc.Graph(id='model_score_graph', figure= model_score_fig)
                    ]),
            ])
        ]
    )
def get_corr_fig(df):
    fig = px.imshow(df.corr(),labels=dict(color='offer_succeed'))
    fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor':'rgba(0, 0, 0, 0)',
            'title': {'text': '1. Correlation Heatmap', 'font': {'color': 'white', 'size': 18}, 'x': 0.5},
            'template': 'plotly_dark',
            })
    return fig
def get_scatter_figure(df):
    fig = px.scatter(df, x = 'income', y = 'difficulty', color='offer_succeed')
    fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor':'rgba(0, 0, 0, 0)',
            'title': {'text': '2. Scatter Plot base on offer_succeed', 'font': {'color': 'white', 'size': 18}, 'x': 0.5},
            'template': 'plotly_dark',
            })
    return fig

def get_model_score_figure():
    models = loadModels()
    model_scores = getModelScore(models, X_test, y_test)
    model_score_df = pd.DataFrame(model_scores, columns =['model', 'accuracy', 'f1'])
    fig = go.Figure()
    fig.add_trace(go.Bar(
                    x=model_score_df['model'],
                    y=model_score_df['accuracy'],
                    name='Accuracy'))
    fig.add_trace(go.Scatter(
                    x=model_score_df['model'],
                    y=model_score_df['f1'],
                    name='F1'))
    fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor':'rgba(0, 0, 0, 0)',
            'title': {'text': '4. Model Score Comparasion Graph', 'font': {'color': 'white', 'size': 18}, 'x': 0.5},
            'template': 'plotly_dark',
            })
    return fig

def get_roc_figure():
    y_test_pred_proba = best_model.predict_proba(X_test)[::,1]
    y_train_pred_proba = best_model.predict_proba(X_train)[::,1]

    fpr_train, tpr_train, _, auc_train, best_fpr_train, best_tpr_train = get_roc_curve(y_train, y_train_pred_proba)
    fpr_test, tpr_test, _, auc_test, best_fpr_test, best_tpr_test = get_roc_curve(y_test, y_test_pred_proba)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('ROC Curve Train'))

    fig.add_trace(
        go.Scatter(x=fpr_train, y=tpr_train, name='train_roc_curve'),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    mode='lines',
                    line=dict(color='firebrick', width=4,
                        dash='dash')),row=1, col=1)

    fig.add_trace(go.Scatter(x=[best_fpr_train], y=[best_tpr_train],
                    mode='markers+text',
                    textposition="bottom right",),row=1, col=1)

    fig.add_trace(
        go.Scatter(x=fpr_test, y=tpr_test, name='test_roc_curve'),
        row=1, col=2
    )
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    mode='lines',
                    line=dict(color='firebrick', width=4,
                              dash='dash')),row=1, col=2)
    fig.add_trace(go.Scatter(x=[best_fpr_test], y=[best_tpr_test],
                    mode='markers+text',
                    textposition="bottom right",),row=1, col=2)

    # Update xaxis properties
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
    fig.update_traces(showlegend=False)
    fig.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor':'rgba(0, 0, 0, 0)',
                    'title': {'text': '3. ROC Curves Graph', 'font': {'color': 'white', 'size': 18}, 'x': 0.5},
                    'template': 'plotly_dark',
                    })
    return fig
# Callback for interactive scatterplot
@app.callback(Output('scatter_graph', 'figure'),
              [Input('selector_x', 'value'), Input('selector_y', 'value')])    
def update_scatterplot(selector_x, selector_y):
    fig = px.scatter(df, x=selector_x, y=selector_y, color='offer_succeed')
    fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor':'rgba(0, 0, 0, 0)',
            'title': {'text': '2. Scatter Plot base on offer_succeed', 'font': {'color': 'white', 'size': 18}, 'x': 0.5},
            'template': 'plotly_dark',
            })
    return fig    

# Route API
@server.route('/')
def show_dashboard():
    return redirect('/dashboard')

@server.route('/api/evaluate/<model_name>', methods=['GET', 'POST'])
def evaluate_model(model_name):
    model = loadModel(model_name)
    result = get_evaluation_report(X_test, y_test, model)
    return jsonify({"evaluate_result": result})

@server.route('/api/predict/<model_name>', methods=['GET', 'POST'])
def predict(model_name):
    model = loadModel(model_name)
    result = get_predict_result(X_test, model)
    return jsonify({"predict_result": list(result)})


@server.route('/api/predict_offer_effective', methods=['POST'])
def predict_offer_effective():
    json_data = request.get_json()
    cust_id = json_data["customer_id"]
    offer_id = json_data["offer_id"]
    time = json_data["time"]
    amount_reward = json_data["amount_reward"]

    result = predict_offer_success(cust_id, offer_id, best_model, time, amount_reward)
    return jsonify({"success": result})

# Run the app
if __name__ == '__main__':
    create_app()
    app.run_server(debug=True, dev_tools_ui=False, dev_tools_props_check=True)