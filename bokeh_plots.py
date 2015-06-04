__author__ = 'miguel'

from bokeh.plotting import *



def make_outliers_ds(df, start=None, end=None, coordinates='absolute'):
    """ construct data source """
    r = df
    ds = ColumnDataSource()
    x = np.append(r.index, np.flipud(r.index))
    if coordinates == 'absolute':
        y = np.append(r.q1 - r.neutral, np.flipud(r.q99 - r.neutral))
    else:
        y = np.append(r.q1 - r.neutral, np.flipud(r.q99 - r.neutral))
    ds.add(x, 'x')
    ds.add(y, 'y')
    ds.add(r.index[:-2], 'date')
    ds.add(r.realized.values[:-2] - r.neutral.values[:-2], 'real')
    return ds

def make_shortfall_ds(df, start=None, end=None):
    ds = ColumnDataSource()
    shortfall_values = df.shortfall.values[:-2]
    ds.add(df.index[:-2], 'date')
    ds.add(np.clip(shortfall_values, 0., np.infty), 'shortfall')
    ds.add(np.clip(shortfall_values, 0., np.infty) * 0.5, 'shortfall_y')
    return ds

def make_outliers_plot(source=None):
    p = figure(title="simple line example", plot_height=400, plot_width=900, x_axis_type="datetime")
    p.line('date', 'real', source=ds, line_width=1, color='#444444')
    p.patch('x', 'y', source=ds, color='grey', fill_alpha=0.4)
    return p

def create_outliers(source,original_p=None):
    x_column = 'date'
    h_column = 'shortfall'
    y_column = 'shortfall_y'
    w = 20*60*60*1000 # half day in ms
    if original_p is None:
        p = figure(title="x", plot_width=900, plot_height=300, x_axis_type="datetime") #, x_range=range_obj, tools=[])
    else:
        p = figure(title="x", plot_width=900, plot_height=300, x_axis_type="datetime", x_range=original_p.x_range, tools=[])
    p.rect(x=x_column, y=y_column, width=w, height=h_column, source=source)
#     p.line(x=x_column, y=h_column, color='green', source=source)
#     p.tools.append(TapTool(plot=p))
#     p.tools.append(WheelZoomTool(dimensions=['width']))
    return p
