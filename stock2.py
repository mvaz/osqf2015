"""
This file demonstrates a bokeh applet, which can be viewed directly
on a bokeh-server. See the README.md file in this directory for
instructions on running.
"""

import logging

logging.basicConfig(level=logging.DEBUG)

import numpy as np

from bokeh.plotting import figure
from bokeh.models import Plot, ColumnDataSource, DataRange1d, TapTool, WheelZoomTool, Circle
from bokeh.charts import Bar
from bokeh.models import BlazeDataSource
from bokeh.properties import Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, VBox, Slider, DateRangeSlider, VBoxForm

import blaze as bz
import bcolz
import pandas as pd

file_name = "notebooks/db2.bcolz"


class StockModel(object):
    """docstring for StockModel"""
    def __init__(self):
        super(StockModel, self).__init__()
        file_name = "notebooks/db2.bcolz"
        self.df = bz.odo(file_name, pd.DataFrame) #[1000:1100]
        self.source = ColumnDataSource(self.df[['Date', 'Close']])
        self.source.add([], 'LogReturns')
        self.source.add([], 'mids')
        self.source.add([], 'Vola')
        self.source.add([], 'ewma')
        self.compute_model()
        # self.source = ColumnDataSource(self.df)
    
    def compute_model(self, _lambda=0.06, n_days=1):
        _com = (1 - _lambda) / _lambda
        log_returns = np.log(self.df.Close.pct_change(periods=n_days) + 1)
        self.source.data['LogReturns'] = log_returns[3:]
        self.source.data['mids'] = log_returns[3:] / 2.
        self.source.data['Vola'] = pd.ewmstd( log_returns, com=_com, ignore_na=True)[3:]
        self.source.data['ewma'] = pd.ewma( log_returns, com=_com, ignore_na=True)[3:]
        print(self.source.data['Vola'])


def devol(log_returns, λ=0.06):
    _com = (1 - λ) / λ
#     log_returns = compute_logreturns(series)
    σ = pd.ewmstd( log_returns, com=_com, ignore_na=True)
    ξ = log_returns / σ
    return ξ, σ
        # self.df['Returns'].

    def devol(self, l=0.96):
        pass

class StockApp(VBox):
    """An example of a browser-based, interactive plot with slider controls."""

    extra_generated_classes = [["StockApp", "StockApp", "VBox"]]

    dates_slider = Instance(DateRangeSlider)
    lambda_slider = Instance(Slider)

    plot = Instance(Plot)
    outliers = Instance(Plot)
    returns = Instance(Plot)

    # histogram_source = Instance(ColumnDataSource)

    # model = Instance(StockModel)
    outliers_source = Instance(ColumnDataSource)

    @classmethod
    def create_histogram(cls, source):
        toolset = "crosshair,pan,reset,resize,save,wheel_zoom,tap"

        # Generate a figure container
        plot = figure(title_text_font_size="12pt",
                      plot_height=400,
                      plot_width=650,
                      tools=toolset,
                      title="Histogram",
                      # x_range=[0, 4*np.pi],
                      # y_range=[-2.5, 2.5]
        )

        # Plot the line by the x,y values in the source property
        plot.quad(top='top', bottom=0, left='left', right='right',
            fill_color="#036564", line_color="#033649",
            alpha=0.5,
            source=source
        )
        return plot

    @classmethod
    def create_outliers(cls, source):
        x_column = "Date"
        y_column = "Vola"

        plot1 = figure(title="Outliers", plot_width=650, plot_height=400, x_axis_type="datetime")
        plot1.tools.append(TapTool(plot=plot1))
        plot1.line(x=x_column, y=y_column, size="size", line_color="red", source=source)
        return plot1

    @classmethod
    def create_returns(cls, source, range_obj):
        p = figure(title="Returns", plot_width=650, plot_height=400, x_axis_type="datetime", x_range=range_obj, tools=[])
        w = 20*60*60*1000 # half day in ms
        p.rect(x="Date", y='mids', width=w, height='LogReturns', source=source)
        p.tools.append(TapTool(plot=p))
        p.tools.append(WheelZoomTool(dimensions=['width']))
        return p



    @classmethod
    def create(cls):
        """One-time creation of app's objects.

        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        obj = cls()

        N = 1000
        # x = np.linspace(-2, 2, N)
        x = np.linspace(0,10,N)
        y = np.sin(x**2)


        model = StockModel()
        obj.outliers_source = model.source

        obj.lambda_slider = Slider(
            title="lambda",
            name='lambda',
            value=0.06,
            start=0.01,
            end=0.99,
            step=0.01
        )
        # start = '2010-01-01'
        # end = '2015-01-01'
        # obj.dates_slider = DateRangeSlider(title="Period:", name="period", value=(start,end), bounds=(start,end), range=(dict(days=1), None))
        # print(obj.dates_slider)


        # Generate a figure container
        obj.outliers = cls.create_outliers(model.source)
        obj.returns = cls.create_returns(model.source, obj.outliers.x_range)
        

        obj.update_data()
        obj.children.append(obj.outliers)
        obj.children.append(obj.returns)
        # obj.children.append(obj.dates_slider)
        # obj.children.append(obj.lambda_slider)   
        # obj.children.append(obj.plot)

        return obj

    def setup_events(self):
        """Attaches the on_change event to the value property of the widget.

        The callback is set to the input_change method of this app.
        """
        super(StockApp, self).setup_events()
        if not self.dates_slider:
            return

        # Slider event registration
        # self.dates_slider.on_change('value', self, 'input_change')
        self.outliers_source.on_change('selected', self, 'on_selection_change')
        # for w in ["bins"]:
        #     getattr(self, w).on_change('value', self, 'input_change')

    def input_change(self, obj, attrname, old, new):
        """Executes whenever the input form changes.

        It is responsible for updating the plot, or anything else you want.

        Args:
            obj : the object that changed
            attrname : the attr that changed
            old : old value of attr
            new : new value of attr
        """
        self.update_data()

    def update_data(self):
        """Called each time that any watched property changes.

        This updates the sin wave data with the most recent values of the
        sliders. This is stored as two numpy arrays in a dict into the app's
        data histogram_source property.
        """
        N = 200
        # Get the current slider values
        l = self.lambda_slider.value
        # obj.model.compute_model(_lambda=l)
        # values = self.histogram_source.data['values']
        #x = self.outliers_source.data['x']
        #y = self.outliers_source.data['y']
        #size = self.outliers_source.data['size']
        
        # hist, edges = np.histogram(values, density=True, bins=bins)

        logging.debug(
            "PARAMS: bins: %d", l
        )

        # self.histogram_source.data['top'] = hist
        # self.histogram_source.data['left'] = edges[:-1]
        # self.histogram_source.data['right'] = edges[1:]
        # self.histogram_source.data = dict(top=hist, bottom=0, left=edges[:-1], right = edges[1:], values=values)
    
    def on_selection_change(self, attr, _, inds, x):
        # print(x, attr, _, inds, x)
        logging.debug(
            "I am being called"
        )


        color = ["blue"] * self.outliers_source.data['N']
        if inds:
            [index] = inds
            color[index] = "red"
            # self.update_histogram_source(index)
            self.update_data()
            # print( index)

        # obj.source.data["color"] = color
        # obj.source.data = dict(top=hist, bottom=0, left=edges[:-1], right = edges[1:], values=values)
        # session.store_objects(source2)

    # def update_histogram_source(self, t):
    #     self.histogram_source.data['values'] = list( map( lambda v: v * t / 10., self.histogram_source.data['values'] ) )

    @classmethod
    def update_outlier_source(self, source, t):
        pass





# The following code adds a "/bokeh/sliders/" url to the bokeh-server. This
# URL will render this sine wave sliders app. If you don't want to serve this
# applet from a Bokeh server (for instance if you are embedding in a separate
# Flask application), then just remove this block of code.
@bokeh_app.route("/bokeh/stock/")
@object_page("Stock")
def make_var():
    app = StockApp.create()
    return app
