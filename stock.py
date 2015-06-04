"""
This file implements a bokeh applet for returns distribution.
It requires a bokeh-server.
See the README.md file in this directory for instructions on running.
"""


import logging

logging.basicConfig(level=logging.DEBUG)

import numpy as np

from bokeh.plotting import figure
from bokeh.models import Plot, ColumnDataSource, DataRange1d, TapTool, Circle
from bokeh.properties import Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, VBox, Slider, TextInput, VBoxForm


class StockApp(VBox):
    """ Applet for exploring the returns of a stock """
    extra_generated_classes = [['StockApp', 'StockApp', 'VBox']]
    stock_plot = Instance(Plot)
    source = Instance(ColumnDataSource)

    @classmethod
    def create_stock(cls, source):

        # xdr1 = DataRange1d(sources=[source.columns("x")])
        # ydr1 = DataRange1d(sources=[source.columns("y")])

        # plot1 = figure(title="Outliers", x_range=xdr1, y_range=ydr1, plot_width=650, plot_height=400)
        stock_plot = figure(title="", plot_width=650, plot_height=400)
        # stock_plot.tools.append(TapTool(plot=stock_plot))
        # stock_plot.line(x="x", y="values", size=12, color="blue", line_dash=[2, 4], source=source)
        return stock_plot
        # plot1.scatter(x="x", y="y", size="size", fill_color="red", source=source)


    @classmethod
    def create(clz):
        """One-time creation of app's objects.

        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        self = clz()
        n_vals = 1000
        self.source = ColumnDataSource(
            data=dict(
                top=[],
                bottom=0,
                left=[],
                right=[],
                x= np.arange(n_vals),
                values= np.random.randn(n_vals)
                ))

        # Generate a figure container
        self.stock_plot = clz.create_stock(self.source)
        self.update_data()
        self.children.append(self.stock_plot)

    def setup_events(self):
        """Attaches the on_change event to the value property of the widget.

        The callback is set to the input_change method of this app.
        """
        super(StockApp, self).setup_events()

        logging.debug("%s" % str(self.source))
        # Slider event registration
        # self.source.on_change('selected', self, 'on_selection_change')
        print("+++++++++++++++++++++++++++++++++")
        print(self)
        self.stock_plot.on_change('value', self, 'input_change')
        # self.outliers_source.on_change('selected', self, 'on_selection_change')
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
        logging.debug("update_data")
        n_vals = 1000
        self.source.data = dict(top=hist, bottom=0, left=0, right = 0, x=np.arange(n_vals), values=np.random.randn(n_vals))

    def on_selection_change(self, attr, _, inds, x):
    	pass

# The following code adds a "/bokeh/sliders/" url to the bokeh-server. This
# URL will render this sine wave sliders app. If you don't want to serve this
# applet from a Bokeh server (for instance if you are embedding in a separate
# Flask application), then just remove this block of code.
@bokeh_app.route("/bokeh/stock/")
@object_page("StockApp")
def make_stock():
    logging.debug('creating')
    app = StockApp.create()
    logging.debug('created')
    return app
