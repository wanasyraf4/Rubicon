import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
import json


# =============================================
# Assuming you have your data prepared
NDVI8 = np.genfromtxt("NDVI8.csv", delimiter=",")
NDVI11 = np.genfromtxt("NDVI11.csv", delimiter=",")
NDVI12 = np.genfromtxt("NDVI12.csv", delimiter=",")

NDVI_value = np.genfromtxt("NDVI_value.csv", delimiter=",")
EVI_value = np.genfromtxt("EVI_value.csv", delimiter=",")



# =================== chart area for disease type =====================


# Define your data
labels = ['South American Leaf Blight (SALB)',
          'Corynespora leaf fall',
          'Anthracnose(PSD)']
values = [6, 4, 2]

# Create the donut plot
figDType = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
                             textinfo='label+percent+value',
                             insidetextorientation='radial',
                             textposition='outside')])

# Add a title
figDType.update_layout(
    annotations=[dict(text='Disease <br> Type', x=0.5, y=0.5, font_size=15, showarrow=False)],
    showlegend=False,
    title_text="Disease Distribution"
)

# fig.show()


# ================ disease severity ===========================

# Define your data
diseases = ['South American Leaf Blight (SALB)', 'Corynespora leaf fall', 'Anthracnose(PSD)']
severity_stages = [1, 2, 3, 4, 5]
reports = [[3, 2, 1, 0, 0], [3, 1, 0, 0, 0], [1, 1, 0, 0, 0]]  # number of reports for each disease at each severity stage

# Create a trace for each disease
traces = []
for i, disease in enumerate(diseases):
    traces.append(go.Scatter(
        x=[disease]*len(severity_stages),  # repeat the disease name for each severity stage
        y=severity_stages,  # severity stages
        mode='markers',
        marker=dict(
            size=reports[i],  # size of bubbles based on number of reports
            sizemode='area',  # size represents area of bubble
            sizeref=2.*max(reports[i])/(40.**2),  # scale the size of the bubbles
            sizemin=4  # minimum size so bubbles are visible even for small number of reports
        ),
        name=disease  # name of the traces/diseases
    ))

# Create the bubble plot
fig_sev = go.Figure(data=traces)

# Set title and labels
fig_sev.update_layout(
    title='Disease Severity',
    xaxis_title='Disease',
    yaxis_title='Severity Stage',
    showlegend=False
)

# fig.show()



# ==================== fig line ==============================================

# NDVIs data
x_data = np.arange(1,len(NDVI8)+1)
y_data = NDVI8

x_data2 = np.arange(1,len(NDVI11)+1)
y_data2 = NDVI11

x_data3 = np.arange(1,len(NDVI12)+1)
y_data3 = NDVI12

# Creating the line plot
fig_line = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='lines', name='Chl absorption max.'))

# Add another trace to the figure
fig_line.add_trace(go.Scatter(x=x_data2, y=y_data2, mode='lines', name='biogeochemistry'))
fig_line.add_trace(go.Scatter(x=x_data3, y=y_data3, mode='lines', name='vegetation'))

# Add title and labels
fig_line.update_layout(title='NDVIs Plot', xaxis_title='Spectral wavelength (nm)', yaxis_title='Vegitation Index')

# Display the figure
# fig_line.show()


# =================== fig area =================================================

# NDVI_value and EVI_value data
x_dataA = np.arange(1,len(NDVI_value)+1)
y_dataA = NDVI_value

x_dataB = np.arange(1,len(EVI_value)+1)
y_dataB = EVI_value

# Creating the area plot
fig_area = go.Figure(data=go.Scatter(x=x_dataA, y=y_dataA, mode='lines', fill='tozeroy', name='NDVI'))

# Add another trace to the figure
fig_area.add_trace(go.Scatter(x=x_dataB, y=y_dataB, mode='lines', fill='tozeroy', name='EVI'))

# Add title and labels
fig_area.update_layout(title='NDVI and EVI Plot', xaxis_title='Spectral wavelength (nm)', yaxis_title='Vegetation Index')

# Display the figure
# fig_area.show()


# ======================== FFT area ======================

# Calculate FFT
fft_result = np.fft.fft(NDVI8)

# Prepare data for plotting
# x_values = range(len(NDVI8))
y_values_real = [val.real for val in fft_result]
y_values_imag = [val.imag for val in fft_result]
xr_values = np.arange(len(y_values_real))
xi_values = np.arange(len(y_values_imag))

AA = np.abs(np.array(y_values_imag) - np.array(y_values_real))
AA_values = np.arange(len(AA))
ksi = np.mean(AA) + np.std(AA)

# Define the lambda function
f = lambda x: 0 if x < ksi else x

# Vectorize the function so it can be applied element-wise to a numpy array
vfunc = np.vectorize(f)

# Apply the function to AA
AA_normalised= vfunc(AA)
AA_normalised_values = np.arange(len(AA_normalised))

# Create a trace for the real part
trace_real = go.Scatter(
    x = xr_values,
    y = y_values_real,
    mode = 'lines',
    name = 'Real part'
)

# Create a trace for the imaginary part
trace_imag = go.Scatter(
    x = xi_values,
    y = y_values_imag,
    mode = 'lines',
    name = 'Imaginary part'
)

# Create a trace for the absolute
trace_absolute = go.Scatter(
    x = AA_values,
    y = AA,
    mode = 'lines',
    name = 'absolute'
)

# Create a trace for the absolute
trace_norm = go.Scatter(
    x = AA_normalised_values,
    y = AA_normalised,
    mode = 'lines+markers',
    name = 'norm'
)

# Create the layout
layout = go.Layout(
    title = 'FFT Analysis on Disease Severity',
    xaxis = dict(title = 'Frequency'),
    yaxis = dict(title = 'Amplitude')
)

# Compile the data
data = [trace_real, trace_imag, trace_absolute, trace_norm]

# # Create the plot
V = go.Figure(data=data, layout=layout)

# # Show the plot
# V.show()

# =========================== disease area =================

# Count the number of non-zero elements in the array
num_nonzero = np.count_nonzero(AA_normalised)

fig_disease = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = num_nonzero,
    title = {'text': "Estimated Number of Diseased Trees"},
    gauge = {'axis': {'range': [None, len(AA_normalised)]}}
))

# fig_disease.show()



# ============================== front end area ======================

app = dash.Dash(__name__)
#Initiate the app 
server = app.server


app.layout = html.Div([
    html.Div([
        html.Img(src="/assets/logo.png", style={'height':'150px', 'display': 'block', 'margin': 'auto'}),
        html.Br(),
        # html.H1("Ribicon, Precision Plantation Data Story", style={'text-align': 'center'})
        html.H1("Rubicon", style={'text-align': 'center', 'font-family': 'Aharoni'}),
        html.Br(),
        html.H1("Precision Plantation Data Story", style={'text-align': 'center'})
    ]),
    
    html.Iframe(srcDoc=open("map.html", "r").read(), width="100%", height="600"),
    html.Div([
        html.Img(src='/assets/colorbar.png', style={'height':'80px', 'width':'55%', 'display': 'block', 'margin': 'auto'})
    ]),
    
    html.Div([
        dcc.Graph(figure=fig_disease, style={"width": "50%", "display": "inline-block"}),
        dcc.Graph(figure=figDType, style={"width": "50%", "display": "inline-block"})
    ]),
    
    
    html.Div([
        dcc.Graph(figure=V, style={"width": "50%", "display": "inline-block"}),
        dcc.Graph(figure=fig_sev, style={"width": "50%", "display": "inline-block"})
    ]),
    
    html.Div([
        dcc.Graph(figure=fig_area, style={"width": "50%", "display": "inline-block"}),
        dcc.Graph(figure=fig_line, style={"width": "50%", "display": "inline-block"})
        
    ]),
])




if __name__ == '__main__':
    app.run_server(debug=True)
