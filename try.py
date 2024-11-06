import plotly.graph_objects as go
import numpy as np
import pickle

with open('/scratch/anoushkrit.scee.iitmandi/github/fusePCD/outputs/A ToothBrush.pkl','rb') as f:
    pc = pickle.load(f)
    
pc = pc[2].cpu()
pc = pc.numpy()
d = pc
# Generate sample data
n = 100
x = d[:, 0]
z = d[:, 1]
y = d[:, 2]  # Random z values
colors = np.random.rand(n)    # Color by another variable

# Create a 3D scatter plot with color mapping
fig = go.Figure(data=go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=colors,  # Color by another variable
        colorscale='Rainbow',  # Color scale (you can use others like 'Viridis', 'Cividis', etc.)
        colorbar=dict(title='Color Scale'),  # Color bar title
        opacity=0.8,
        line=dict(width=0.5)
    ),
    name='Data Points'
))

# Add title and labels
fig.update_layout(
    title='3D Scatter Plot with Color Mapping',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Show the plot
fig.write_html('x.html')