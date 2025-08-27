# 1. Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# 2. Initialize the Dash app
app = dash.Dash(__name__)

# 3. Create Sample Yearly Data
# This data structure is designed to be filtered by the 'Metric' column.
data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2018, 2019, 2020, 2021, 2022, 2018, 2019, 2020, 2021, 2022],
    'Metric': ['Revenue', 'Revenue', 'Revenue', 'Revenue', 'Revenue',
               'Profit', 'Profit', 'Profit', 'Profit', 'Profit',
               'New Customers', 'New Customers', 'New Customers', 'New Customers', 'New Customers'],
    'Value': [120, 150, 130, 180, 210, 30, 40, 25, 50, 65, 5000, 6500, 5800, 7200, 8100]
}
df = pd.DataFrame(data)

# 4. Define the app layout
app.layout = html.Div(children=[
    html.H1(
        children='Yearly Report Statistics Dashboard',
        style={'textAlign': 'center'}
    ),

    html.Label('Select a Report Type:'),
    dcc.Dropdown(
        id='report-type-dropdown',
        options=[
            {'label': 'Revenue (in Millions)', 'value': 'Revenue'},
            {'label': 'Profit (in Millions)', 'value': 'Profit'},
            {'label': 'New Customers Acquired', 'value': 'New Customers'}
        ],
        value='Revenue' # Set the default value
    ),

    # This Graph component will be updated by the callback
    dcc.Graph(id='yearly-report-graph')
])

# 5. Define the Callback Function
@app.callback(
    Output('yearly-report-graph', 'figure'),
    Input('report-type-dropdown', 'value')
)
def update_yearly_graph(selected_report_type):
    # This function is triggered whenever the dropdown value changes.
    
    # Filter the DataFrame based on the user's selection
    filtered_df = df[df['Metric'] == selected_report_type]

    # Create a new bar chart with the filtered data
    fig = px.bar(
        filtered_df,
        x='Year',
        y='Value',
        title=f'Yearly Trend for {selected_report_type}',
        labels={'Value': selected_report_type}
    )

    # Return the newly created figure to the 'figure' property of the dcc.Graph
    return fig

# 6. Run the app
if __name__ == '__main__':
    app.run(debug=True)