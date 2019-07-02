import plotly
import plotly.plotly as py
from plotly.graph_objs import graph_objs
from IPython.display import SVG, display, Image

# set credentials
plotly.tools.set_credentials_file(username='gium', api_key='nUFs5UnmuBR3pEbGIMj8')

def get_layout(_xlab="",_ylab="",_title=""):
    """Format xlab, ylab and title
    
    Args:
        _xlab (str): label of x-axis
        _ylab (str): label of y-axis
        _title (str): plot title
        
    Returns:
        Layout object
    """
    layout = graph_objs.Layout(
                xaxis=dict(
                    title=_xlab,
                     titlefont=dict(
                        family='Arial',
                        size=18,
                        color='#7f7f7f'
                    ),
                    tickangle=45,
                    zeroline = False

                ),
                yaxis=dict(
                    title=_ylab,
                     titlefont=dict(
                        family='Arial',
                        size=18,
                        color='#7f7f7f'
                    ),
                    zeroline = False
            #         range=[0, 100000]

                ),
                height=600,
                margin=graph_objs.layout.Margin(
                    l=50,
                    r=50,
                    b=250,
                    t=100,
                    pad=4
                ),
                title = _title
            )
    return layout

def show_img(path):
    """Rander svg/png image into jupyter-lab notebook
    """
    extension = path.split(".")
    extension = extension[-1]
    if extension == 'svg': display(SVG(filename=path))
    else: 
        img = Image(path)
        display(img)