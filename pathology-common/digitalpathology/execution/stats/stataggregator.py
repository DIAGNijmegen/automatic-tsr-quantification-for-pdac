"""
This file contains a class for statistics handling and plotting.
"""

import pandas as ps
import numpy as np
import os
import matplotlib

# Set the backend to Agg to write to PNG files without GUI support of the operating system. Unfortunately there is no nicer way to do this. The interactive backends crash upon
# importing pyplot, before they could be switched with a non-interactive one. The other way would be to configure the default backend, but that would need the editing of the
# matplotlibrc file and make the system depend on an external setting.
#
matplotlib.use('agg')

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------

class StatAggregator(object):
    """This class can collect, save and plot statistics from learning procedure of deep neural networks."""

    def __init__(self, epoch_save_path, epoch_plot_path, epoch_stats_to_plot, epoch_vertical_lines, experiment_name, append):
        """
        Initialize the object.

        Args:
            epoch_save_path (str, None): Epoch statistics save file path.
            epoch_plot_path (str, None): Epoch statistics plot file path.
            epoch_stats_to_plot (list): List of statistics to plot.
            epoch_vertical_lines (list): List of statistics to draw vertical lines, paired with bool indicating on increase.
            experiment_name (str): Experiment name for plotting.
            append (bool): Whether to overwrite existing statistics files instead of appending.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__epoch_save_path = epoch_save_path                  # Path for saving the epoch statistics.
        self.__epoch_plot_path = epoch_plot_path                  # Path for plotting progress.
        self.__epoch_data_frame = None                            # Data frame for epoch statistics.
        self.__append = append                                    # Store whether to overwrite or append.
        self.__experiment_name = experiment_name                  # Experiment name for plotting.
        self.__epoch_stats_to_plot = list(epoch_stats_to_plot)    # List of statistics to plot.
        self.__epoch_vertical_lines = list(epoch_vertical_lines)  # Dictionary of statistics for vertical lines.

        # Hardcoded values.
        #
        self.__plot_epoch_tick_count = 100  # Epoch tick count for plotting.
        self.__plot_value_tick_freq = 0.05  # Value print line frequency for plotting.
        self.__plot_dpi = 100               # Plot render DPI.
        self.__plot_size = (4000, 2000)     # Plot target image size.

    def __plotdataset(self, target_path, data_set, vertical_lines, x_ticks, x_label, y_range):
        """
        Plot the collected data set.

        Args:
            target_path (str): Target plot file path.
            data_set (list): Data set to plot. List of (legend text, data array) pairs.
            vertical_lines (list): Vertical line indices. List of (legend text, data array) pairs.
            x_ticks (list): X tick labels.
            x_label (str): Label of X axis.
            y_range (tuple): Value range to plot in in (min, max) tuple format.
        """

        # Configure target resolution.
        #
        plt.figure(figsize=(self.__plot_size[0] // self.__plot_dpi, self.__plot_size[1] // self.__plot_dpi), dpi=self.__plot_dpi)

        # Plot the collected data.
        #
        if self.__experiment_name:
            plt.title(self.__experiment_name)

        fig, ax = plt.subplots()
        for line_name, line_indices in vertical_lines:
            next_color = ax._get_lines.get_next_color()
            plot_line_name = line_name
            for line_index in line_indices:
                if plot_line_name:
                    plt.axvline(x=line_index, linestyle=':', label=plot_line_name, color=next_color)
                    plot_line_name = None
                else:
                    plt.axvline(x=line_index, linestyle=':', color=next_color)

        item_indices = list(range(len(x_ticks)))
        for plot_item in data_set:
            plt.plot(item_indices, plot_item[1], label=plot_item[0])

        # Calculate ticks.
        #
        if len(x_ticks) < self.__plot_epoch_tick_count:
            plot_x_indices = item_indices
            plot_x_ticks = x_ticks
        else:
            plot_x_step = len(x_ticks) / self.__plot_epoch_tick_count
            plot_x_indices = [round(index * plot_x_step) for index in range(self.__plot_epoch_tick_count)]
            plot_x_ticks = [x_ticks[index] for index in plot_x_indices]

            if plot_x_ticks[-1] != x_ticks[-1]:
                plot_x_indices.append(len(x_ticks) - 1)
                plot_x_ticks.append(x_ticks[-1])

        plot_y_indices = np.arange(y_range[0], y_range[1] + self.__plot_value_tick_freq, self.__plot_value_tick_freq)

        # Configure the legend and the ticks.
        #
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.xlabel(x_label)
        plt.xticks(plot_x_indices, plot_x_ticks, rotation=45)
        plt.yticks(plot_y_indices)
        plt.grid(True)

        # Save the generated figure and clean up.
        #
        plt.savefig(target_path, dpi=self.__plot_dpi, bbox_inches='tight')
        plt.close()

    def addplots(self, stat_names):
        """
        Add statistics name to the listed of plotted items.

        Args:
            stat_names (list): List of stat names.
        """

        # Go through the names and add the ones that are not present already.
        #
        for stat_name_item in stat_names:
            if stat_name_item not in self.__epoch_stats_to_plot:
                self.__epoch_stats_to_plot.append(stat_name_item)

    def addlines(self, vertical_lines):
        """
        Add vertical line to plot.

        Args:
            vertical_lines (list, tuple): List or a single tuple of statistics name and draw on increase flags.
        """

        # Go through the line configurations and add the ones that are not present already.
        #
        for vertical_line_item in vertical_lines:
            if vertical_line_item not in self.__epoch_vertical_lines:
                self.__epoch_vertical_lines.append(vertical_line_item)

    def load(self):
        """Load the existing statistics file."""

        # Read the previous statistics file if appending is configured.
        #
        if self.__append and self.__epoch_save_path is not None and os.path.isfile(self.__epoch_save_path):
            self.__epoch_data_frame = ps.read_csv(self.__epoch_save_path)

    def append(self, epoch_statistics_row):
        """
        Append data to the epoch data frame.

        Args:
            epoch_statistics_row (dict): Data row. Column name to value mapping.
        """

        if self.__epoch_data_frame is not None:
            self.__epoch_data_frame = self.__epoch_data_frame.append([epoch_statistics_row], ignore_index=True)
        else:
            self.__epoch_data_frame = ps.DataFrame.from_records([epoch_statistics_row])

    def rewind(self, index):
        """
        Remove the entries that are in or after the given epoch making the stats handler ready to receive data from the index-th epoch.

        Args:
            index (int): Index of epoch.
        """

        self.__epoch_data_frame = self.__epoch_data_frame[0:index]

    def save(self):
        """Save the data frames to the configured paths."""

        if self.__epoch_data_frame is not None and self.__epoch_save_path:
            self.__epoch_data_frame.to_csv(self.__epoch_save_path, index=False)

    def plot(self):
        """Plot the epoch statistics."""

        # Check epoch statistics plotting is configured.
        #
        if self.__epoch_data_frame is not None and self.__epoch_plot_path:
            # Build data structure to plot.
            #
            plot_data = [(key, self.__epoch_data_frame[key].values) for key in self.__epoch_stats_to_plot if key in self.__epoch_data_frame]

            plot_lines = []
            for line_key, on_increase in self.__epoch_vertical_lines:
                direction = 'ascent' if on_increase else 'decay'
                line_name = '{name} {change}'.format(name=line_key, change=direction)
                if on_increase:
                    line_indices = [index - 0.5 for index in range(1, len(self.__epoch_data_frame)) if self.__epoch_data_frame.loc[index][line_key] > self.__epoch_data_frame.loc[index - 1][line_key]]
                else:
                    line_indices = [index - 0.5 for index in range(1, len(self.__epoch_data_frame)) if self.__epoch_data_frame.loc[index][line_key] < self.__epoch_data_frame.loc[index - 1][line_key]]
                plot_lines.append((line_name, line_indices))

            if plot_data:
                # Configure axes.
                #
                epoch_ticks = self.__epoch_data_frame.index.values.tolist()
                epoch_range = (0.0, 1.0)

                # Plot the data.
                #
                self.__plotdataset(target_path=self.__epoch_plot_path, data_set=plot_data, vertical_lines=plot_lines, x_ticks=epoch_ticks, x_label='epoch', y_range=epoch_range)

    @property
    def savepath(self):
        """
        Get the epoch statistics save file path.

        Returns:
            (str, None): Epoch statistics save file path.
        """

        return self.__epoch_save_path

    @property
    def plotpath(self):
        """
        Get the epoch statistics plot file path.

        Returns:
            (str, None): Epoch statistics plot file path.
        """

        return self.__epoch_plot_path

