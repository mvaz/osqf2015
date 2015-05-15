"""
This file demonstrates a bokeh applet, which can be viewed directly
on a bokeh-server. See the README.md file in this directory for
instructions on running.
"""

import logging

logging.basicConfig(level=logging.DEBUG)

import numpy as np

from bokeh.plotting import figure, gridplot
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
        self.df = bz.odo(file_name, pd.DataFrame)[['Date', 'Close']] #[1000:1100]
        self.returns_df = None
        self.compute_model()

    def compute_model(self, _lambda=0.06, n_days=1):
        _com = (1 - _lambda) / _lambda
        self.df['LogReturns'] = np.log(self.df.Close.pct_change(periods=n_days) + 1)
        self.df['Vola'] = pd.ewmstd( self.df.LogReturns, com=_com, ignore_na=True)[2:]
        self.df['DevolLogReturns'] = self.df.LogReturns / self.df.Vola
        self.df.set_index('Date', inplace=True)

    def compute_returns(self, x, n_scenarios=750, bins=45):
        dates = pd.to_datetime(x, unit='ms')
        print(x, dates, dates[0], '<<<?>>>>', dates[0].date())
        max_date = dates[0].date()
        min_date = max_date.replace(year=max_date.year-3)
        self.returns_df = self.df[min_date:max_date].ix[-n_scenarios:]
        hist, edges = np.histogram(self.returns_df.DevolLogReturns, density=True, bins=bins)
        return hist, edges

    def compute_data_source(self):
        source = ColumnDataSource(self.df.reset_index()[2:])
        source.add(self.df[2:].LogReturns.ge(0).map(lambda x: "steelblue" if x else "red"), 'LogReturnsColor')
        source.add(self.df[2:].DevolLogReturns / 2., 'y_mids')
        return source

    def compute_histo_source(self, source, hist, edges):
        source.data = dict(top=hist, bottom=0, left=edges[:-1], right = edges[1:])


    # def compute_model(self, _lambda=0.06, n_days=1):
        # log_returns = np.log(self.df.Close.pct_change(periods=n_days) + 1)
        # self.source.data['LogReturns'] = log_returns[2:]
        # self.source.data['LogReturnsColor'] = log_returns[2:].ge(0).map(lambda x: "#75968f" if x else "#a5bab7")
        # self.source.data['mids'] = log_returns[2:] / 2.
        # self.source.data['Vola'] = pd.ewmstd( log_returns, com=_com, ignore_na=True)[2:]
        # self.source.data['ewma'] = pd.ewma( log_returns, com=_com, ignore_na=True)[2:]

model = StockModel()

class StockApp(VBox):
    """An example of a browser-based, interactive plot with slider controls."""

    extra_generated_classes = [["StockApp", "StockApp", "VBox"]]

    dates_slider = Instance(DateRangeSlider)
    lambda_slider = Instance(Slider)

    histogram = Instance(Plot)
    outliers = Instance(Plot)
    returns = Instance(Plot)

    outliers_source = Instance(ColumnDataSource)
    histogram_source = Instance(ColumnDataSource)

    @classmethod
    def create_histogram(cls, source):
        toolset = "crosshair,pan,reset,resize,save,wheel_zoom"

        # Generate a figure container
        plot = figure(title_text_font_size="12pt",
                      plot_height=400,
                      plot_width=650,
                      tools=toolset,
                      title="Histogram",
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
        y_column = "Close"

        plot1 = figure(title="Outliers", plot_width=650, plot_height=400, x_axis_type="datetime")
        plot1.tools.append(TapTool(plot=plot1))
        plot1.line(x=x_column, y=y_column, size="size", line_color="red", source=source)
        return plot1

    @classmethod
    def create_returns(cls, source, range_obj):
        p = figure(title="Returns", plot_width=650, plot_height=400, x_axis_type="datetime", x_range=range_obj, tools=[])
        w = 20*60*60*1000 # half day in ms
        p.rect(x="Date", y='y_mids', width=w, height='DevolLogReturns', color='LogReturnsColor', source=source)
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
        obj.outliers_source = model.compute_data_source()
        obj.histogram_source = ColumnDataSource(
            data=dict(
                top=[],
                bottom=0,
                left=[],
                right=[],
                ))

        obj.lambda_slider = Slider(
            title="lambda",
            name='lambda',
            value=0.06,
            start=0.01,
            end=0.99,
            step=0.01
        )

        # Generate a figure container
        obj.outliers = cls.create_outliers(obj.outliers_source)
        obj.returns = cls.create_returns(obj.outliers_source, obj.outliers.x_range)
        obj.histogram = cls.create_histogram(obj.histogram_source)
        
        obj.update_data()
        obj.children.append(gridplot([[obj.outliers], [obj.returns, obj.histogram]]))

        return obj

    def setup_events(self):
        """Attaches the on_change event to the value property of the widget.

        The callback is set to the input_change method of this app.
        """
        super(StockApp, self).setup_events()
        if not self.outliers:
            return

        # Registration for selection event
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
        
        

        logging.debug(
            "PARAMS: bins: %d", l
        )

        # hist, edges = np.histogram(values, density=True, bins=bins)
        # self.histogram_source.data['top'] = hist
        # self.histogram_source.data['left'] = edges[:-1]
        # self.histogram_source.data['right'] = edges[1:]
        # self.histogram_source.data = dict(top=hist, bottom=0, left=edges[:-1], right = edges[1:], values=values)
    
    def on_selection_change(self, attr, old, new, x):
        # print(x, attr, _, inds, x)
        logging.debug(
            "I am being called"
        )

        if x:
            inds = np.array(x['1d']['indices'])
            print("......>>> ", inds)
            h = np.take(self.outliers_source.data['Date'], inds)
            print(h)
            hist, edges = model.compute_returns(h)
            model.compute_histo_source(self.histogram_source, hist, edges)
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
