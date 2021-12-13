from xstanpy.base import *
import matplotlib.pyplot as plt

class Plot(Object):
    configuration_names = ('color', 'alpha', 'label', 'marker', 'linestyle')
    def as_ax(self, **kwargs): return Ax([self], configuration=kwargs)

class LinePlot(Plot):
    arg_names = ('x', 'y')

    def update(self, ax):
        if hasattr(self, 'y'):
            ax.plot(self.x, self.y, **self.configuration)
        else:
            ax.plot(self.x, **self.configuration)

class ScatterPlot(LinePlot):
    arg_names = ('x', 'y')
    marker='.'
    linestyle=''

class FillPlot(Plot):
    arg_names = ('x', 'y')

    def update(self, ax):
        if hasattr(self, 'y'):
            x, y = self.x, self.y
        else:
            x, y = np.arange(len(self.x[0])), self.x
        if len(y) != 2:
            y = np.quantile(y, [.159, 1-.159], axis=0)
        ax.fill_between(x, *y, **self.configuration)

class HistPlot(Plot):
    arg_names = ('x', )
    def update(self, ax):
        ax.hist(self.x, **self.configuration)

class VerticalLine(Plot):
    arg_names = ('x', )

    def update(self, ax):
        ax.axvline(self.x, **self.configuration)

class HorizontalLine(Plot):
    arg_names = ('y', )

    def update(self, ax):
        ax.axhline(self.y, **self.configuration)

class Ax(Object):
    arg_names = ('plots', )
    configuration_names = (
        'title',
        'xlabel', 'ylabel',
        'xlim', 'ylim',
        'xscale', 'yscale'
    )

    @cproperty
    def labels(self):
        return tuple([
            plot.label for plot in self.plots if hasattr(plot, 'label')
        ])

    @cproperty
    def show_legend(self): return len(self.labels)
    legend_location = 'best'

    def update(self, ax, **kwargs):
        ax.set(**dict(kwargs, **self.configuration))
        rv = tuple([
            plot.update(ax) for plot in self.plots
        ])
        if self.show_legend:
            ax.legend(loc=self.legend_location)
        return rv

class Figure(Object):
    arg_names = ('axes', )
    configuration_names = ('figsize', 'sharex', 'sharey')
    suptitle = ''
    suptitle_configuration = Configuration(dict(
        t='suptitle',
        family='suptitle_family'
    ))
    legend_configuration = Configuration(dict(
        loc='legend_location',
        bbox_to_anchor='legend_bbox',
        title='legend_title'
    ))
    show_legend = False
    layout_configuration = Configuration(('pad', 'rect', ))
    pad = 3
    ax_configuration = Configuration(dict())


    @cproperty
    def axes2d(self): return np.atleast_2d(self.axes)

    @cproperty
    def no_rows(self): return self.axes2d.shape[0]

    @cproperty
    def no_cols(self): return self.axes2d.shape[1]
    col_width = 4
    row_height = 4

    @cproperty
    def legend_width(self):
        return 2 if self.show_legend == 'row' else 0

    @cproperty
    def figsize(self):
        return (
            self.legend_width + self.no_cols * self.col_width,
            self.no_rows * self.row_height
        )

    @cproperty
    def legend_location(self):
        if self.show_legend == 'row': return 'center left'
        raise NotImplementedError()

    @cproperty
    def legend_bbox(self):
        if self.show_legend == 'row': return (1, .5)
        raise NotImplementedError()


    @cproperty
    def rect(self):
        return (
            0,
            0,
            1-self.legend_width/self.figsize[0],
            1
        )

    @cproperty
    def fig(self):
        fig, axes = plt.subplots(
            self.no_rows, self.no_cols,
            squeeze=False,
            **self.configuration
        )
        if self.suptitle:
            fig.suptitle(**self.suptitle_configuration)
        plt.tight_layout(**self.layout_configuration)
        for i in range(self.no_rows):
            for j in range(self.no_cols):
                self.axes2d[i,j].update(axes[i,j], **self.ax_configuration)
            if self.show_legend == 'row':
                axes[i, -1].legend(**self.legend_configuration)
        return fig

    def save(self, path):
        path = pathlib.Path(path)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path)
