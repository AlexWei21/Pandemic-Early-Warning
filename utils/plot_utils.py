import plotly.graph_objects as go

def plot_compartment(tspan, data, final):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tspan, y=data[:, 0], mode='markers', name='Observed Infections', line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=tspan, y=data[:, 1], mode='markers', name='Observed Deaths', line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=tspan, y=final[:, 0], mode='lines+markers', name='Fitted Infections'))
    fig.add_trace(go.Scatter(x=tspan, y=final[:, 1], mode='lines+markers', name='Fitted Deaths'))
    fig.update_layout(title='SEIRD: Observed vs Fitted',
                       xaxis_title='Day',
                       yaxis_title='Counts',
                       title_x=0.5,
                      width=1000, height=600
                     )
    return fig