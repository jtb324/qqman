# -*- coding: utf-8 -*-
import argparse
import traceback
import logging
import numbers
import pandas as pd
import seaborn as sns
import numpy as np
from numpy import where
import os

import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass, field
from typing import Any, TypeVar, Type

from .biostats import ppoints


class IncorrectCmapVarType(Exception):
    """Exception to be raised if the CmapVarType is not an integer"""

    def __init__(self, type) -> None:
        super().__init__(
            f"cmap_var type provided: {type}. Expected type to be either an integer"
        )


class IncorrectCmapInstanceType(Exception):
    """Execption to be raised if the user provides a colormap that is not the LinearSegmentedColormap type"""

    def __init__(self, type) -> None:
        super().__init__(
            f"Colormap type provide: {type}. Expected the type to be a LinearSegmentedColormap or None. Other types may be supported in future releases"
        )


class RequiredColumnsNotFound(Exception):
    """Exception that will be raised if the required columns are not found in the association dataframe"""

    def __init__(self, message) -> None:
        super().__init__(message)


# creating a type for the update_status_cols class method of the BcfOutputIndices class
T = TypeVar("T", bound="ColorMapper")


@dataclass
class ColorMapper:

    # This attribute is a list of hex colors
    color_list: list[str]
    cmap_var: int 

    @classmethod
    def create_mapper(
        cls: Type[T],
        cmap: list[colors.LinearSegmentedColormap] | None = None,
        cmap_var: int = 2,
    ) -> T:
        """Factory method that will create the color list and then return the class

        Parameters
        ----------
        cmap : list[colors.LinearSegmentedColormap] | None
                Either a LinearsegmentedColormap object or None value. If it is none then the program will create a grey
                Colormap object. Defaults to None

        cmap_var : int
                This is an integer that describes how many different colors the user wants

        Returns
        -------
        ColorMapper
                Returns a ColorMapper object that has a color_list attribute that is the list of different colors
        """
        # if the cmap_var is the instance of the wrong type then we raise an error

        if not isinstance(cmap_var, numbers.Number):
            raise IncorrectCmapVarType(type(cmap_var))
        # if the cmap is not the LinearSegmentedColormap then we will raise an error
        if not (isinstance(cmap, colors.LinearSegmentedColormap) or cmap == None):
            raise IncorrectCmapInstanceType(type(cmap))
        # If there is no cmap then we will we just use the Greys_r colormap
        if not cmap:
            cmap = plt.get_cmap("Greys_r")

        print(f"Using the {cmap.name} colormap")
        # creating a list to append colors to
        color_list = []

        match cmap_var:
            # if the cmap var is 2 then we will we will choose two ends of the color list
            case 2:
                color_list = [colors.to_hex(cmap(0)[:3]), colors.to_hex(cmap(80)[:3])]

                return cls(color_list, cmap_var)
            # if the cmap var != 2 then we we create a step and iterate over the colors in the cmap
            case _:
                step = int(256 / cmap_var)
                for i in range(256):
                    if i % step == 0:
                        color_list.append(colors.to_hex(cmap(i)[:3]))
                return cls(color_list, cmap_var)


@dataclass
class Manhattan:
    """Class to manage the creation of the Manhattan plot
    Parameters
    ----------
    colormap : ColorMap
            ColorMap object that has the attribute color_list which is a list of hex colors

    Output : str
            path to write the output to. Default value is '.'

    internal_parameters : dict[str, Any]
            Attribute dictionary that has parameters such as necessary column names
    """

    colormap: ColorMapper
    output: str = "./"
    internal_parameters: dict[str, Any] = field(default_factory=dict)

    def set_cols(
        self,
        chr_col: str = "CHR",
        bp_col: str = "BP",
        pvalue_col: str = "P-value",
        snp_col: str = "SNP",
    ) -> None:
        """function that will add the columns to the internal parameters dictionary"""
        self.internal_parameters["chr_col"] = chr_col
        self.internal_parameters["bp_col"] = bp_col
        self.internal_parameters["pvalue_col"] = pvalue_col
        self.internal_parameters["snp_col"] = snp_col

    def set_plot_params(
        self,
        title: str = "Output_Plot.png",
        show_plot: bool = False,
        xtick_size: int = 10,
        ytick_size: int = 10,
        xrotation: int = 0,
        yrotation: int = 0,
        label_size: int = 15,
        title_size: int = 20,
    ) -> None:
        """Method that will add parameters for the plot configurations

        Parameters
        ----------
        title : str
                title for the plot. Defaults to Output_Plot.png

        show_plot : bool
                Boolean value for if the user wants the plot to be shown or not. Defaults
                to False

        xtick_size : int
                size of the xticks. Defaults to 10

        ytick_size : int
                size of the yticks. Defaults to 10

        xrotations : int
                how far to rotate x labels. Defaults to 0

        yrotation : int
                how far to rotate the y labels. Defaults to 0

        label_size : int
                size of the labels. Defaults to 15

        title_size : int
                Size of the plot title. Defaults to 20
        """
        self.internal_parameters["title"] = title
        self.internal_parameters["show"] = show_plot
        self.internal_parameters["xtick_size"] = xtick_size
        self.internal_parameters["ytick_size"] = ytick_size
        self.internal_parameters["xrotation"] = xrotation
        self.internal_parameters["yrotation"] = yrotation
        self.internal_parameters["label_size"] = label_size
        self.internal_parameters["title_size"] = title_size

    def set_suggestive_lines(self, suggestive: float = -np.log10(1e-5), genomewide: float = -np.log10(5e-8)) -> None:
        """Method that will add parameters for the suggestive line and genome 
        wide significance

        Parameters
        ----------
        suggestive : float
            line for suggestive significance 

        genomewide : float
            line for genomewide significance
        """
        self.internal_parameters["suggestive"] = suggestive
        self.internal_parameters["genomewide"] = genomewide

    def _check_columns(self, assoc_data: pd.DataFrame) -> None:
        """Method that will make sure the necessary columns are in the dataframe
        Parameters
        ----------
        assoc_data : pd.DataFrame
            Pandas dataframe that has the at least three columns for the chromosome,
            the base position, and the pvalue
        """
        necessary_cols: list[str] = [
            self.internal_parameters["chr_col"],
            self.internal_parameters["bp_col"],
            self.internal_parameters["pvalue_col"],
        ]

        df_cols = assoc_data.columns

        missing_cols = [col for col in necessary_cols if col not in df_cols]

        if missing_cols:
            raise RequiredColumnsNotFound(
                f"{len(missing_cols)} columns were missing from the association dataframe. Expected it to have the columns {', '.join(necessary_cols)}"
            )

    @staticmethod
    def _convert_columns(df_assoc: pd.DataFrame, column_name: str, type_conversion: Any) -> None:
        """Staticmethod that will attempt to convert the column to a type
        Parameters
        ----------
        df_assoc : pd.DataFrame
            dataframe that has the results of the association analysis
        
        column_name : str
            column to be converted
        
        type_conversion : Any
            This is the type the user wants to convert the column to. It could be str, 
            int, float, or others
        """
        # we are going to try to convert the column to the right type but it may fail and we need to catch that
        try:
            df_assoc[column_name] = df_assoc[column_name].astype(type_conversion)
        except ValueError as e:
            print(f"There was an error trying to convert the column {column_name} to type {type_conversion}. The traceback is shown below:\n {e}")
    
    @staticmethod
    def _generate_individ_indx(assoc_df: pd.DataFrame, chromo_col: str, weight_gap: int) -> list[int]:
        """Staticmethod that will create an integer list for each variant in 
        the file. It adds a weight to adjust scale of the list

        Parameters
        ----------
        assoc_data : pd.DataFrame
                Pandas dataframe that has the at least three columns for the chromosome, the base position, and the pvalue

        chromo_col : str
            column that has the chromosome values in it.
        Returns
        -------
        list[int]
            list that has indices for each variant plus a weight component for different chromosomes
        """
        list_ind = []

        for chromo, assoc_data in assoc_df.groupby(by=chromo_col):

            if len(list_ind) == 0:

                last_ind = 0
            else:
                last_ind = (list_ind[-1] + 1) + (10 * weight_gap)

            list_ind += [
                    last_ind + num for num in range(assoc_data.shape[0])
                ]
        
        return list_ind


    def plot(self, assoc_data: pd.DataFrame) -> None:
        """Method to plot the Manhattan plot
        Parameters
        ----------
        assoc_data : pd.DataFrame
                Pandas dataframe that has the at least three columns for the chromosome, the base position, and the pvalue
        """
        # first make sure all the necessary columns are present
        self._check_columns(assoc_data)
        # WE are going to convert each column to the appropriate type. A ValueError will be 
        # raised if it fails
        self._convert_columns(assoc_data, self.internal_parameters["chr_col"], "category")
        self._convert_columns(assoc_data, self.internal_parameters["bp_col"], int)
        self._convert_columns(assoc_data, self.internal_parameters["pvalue_col"], float)

        assoc_data.loc[:,"LOG_P"] = -np.log10(assoc_data[self.internal_parameters["pvalue_col"]])
        
        running_pos = 0
        cumulative_pos = []

        assoc_data.loc[:, "chr"] = where(assoc_data.chromosome.str.len() == 4, assoc_data.chromosome.str[-1], assoc_data.chromosome.str[-2:])
        assoc_data = assoc_data.sort_values(by=["chr", "baseposition"])
        assoc_data.reset_index(drop=True, inplace=True)

        for chrom, group_df in assoc_data.groupby(self.internal_parameters["chr_col"]):  
            cumulative_pos.append(group_df[self.internal_parameters["bp_col"]] + running_pos)
            running_pos += group_df[self.internal_parameters["bp_col"]].max()

        assoc_data.loc[:,"cumulative_pos"] = pd.concat(cumulative_pos)

        print(assoc_data)

        plot = sns.relplot(
            data = assoc_data,
            x = 'cumulative_pos',
            y = 'LOG_P',
            aspect = 4,
            hue = "chromosome",
            palette = "Greys_r",
            linewidth=0,
            s=6,
            legend=None,
        )

        plot.ax.set_xlabel('Chromosome')

        plot.ax.set_xticks(assoc_data.groupby('chromosome')['cumulative_pos'].median())

        plot.ax.set_xticklabels(assoc_data['chromosome'].unique())

        plot.fig.suptitle('GWAS plot showing association between SNPs on autosomes and speeding')
        # creating the space for the figure
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        # fig.tight_layout()
        
        # # we are going to create a dictionary that has the counts for how 
        # # many variants each chromosome has
        # variant_counts = {}
        # # iterate over each unique value
        # # for chr_val in assoc_data[self.internal_parameters["chr_col"]].unique():
        # #     variant_counts[chr_val] = assoc_data[assoc_data[self.internal_parameters["chr_col"]] == chr_val].shape[0]
        # # print(variant_counts)
        # # making a weight that corresponds to the smallest number of variants per chromosome
        # weight_gap = int(min(variant_counts.values()) / 100)

        # list_ind = self._generate_individ_indx(assoc_data, self.internal_parameters["chr_col"], weight_gap)
        # print(list_ind)

        # assoc_data["IND"] = list_ind
        # assoc_data["LOG_P"] = -np.log10(assoc_data[self.internal_parameters["pvalue_col"]])

        # x_ticks, x_labels = list(), list()

        # for i, cChr in enumerate(assoc_data[self.internal_parameters["chr_col"]].unique()):
        #     ind = assoc_data[assoc_data[self.internal_parameters["chr_col"]] == cChr]["IND"]
        #     log_p = assoc_data[assoc_data[self.internal_parameters["chr_col"]] == cChr]["LOG_P"]

        #     ax.scatter(
        #         ind, log_p, marker=".", s=5, color=self.colormap.color_list[i % self.colormap.cmap_var]
        #     )
        #     x_ticks.append(ind.iloc[0] + (ind.iloc[-1] - ind.iloc[0]) / 2)
        #     x_labels.append(cChr)

        # x_padding = len(list_ind) / 20
        # xlim_min = list_ind[0] - x_padding
        # xlim_max = list_ind[-1] + x_padding

        # if self.internal_parameters["suggestive"]:
        #     ax.plot([xlim_min, xlim_max], [self.internal_parameters["suggestive"], self.internal_parameters["suggestive"]], "b-")
        # if self.internal_parameters["genomewide"]:
        #     ax.plot([xlim_min, xlim_max], [self.internal_parameters["genomewide"], self.internal_parameters["genomewide"]], "r-")

        # ax.spines["right"].set_visible(False)
        # ax.spines["top"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)

        # ax.set_xticks(x_ticks)
        # ax.set_xticklabels(x_labels)
        # # ax.set_ylim(bottom=0,top=math.floor(1+max(df_assoc["LOG_P"])+(max(df_assoc["LOG_P"])/100)))
        # # ax.set_ylim(bottom=0,top=max(df_assoc["LOG_P"]))
        # ax.set_ylim(bottom=0)
        # ax.set_xlim([xlim_min, xlim_max])

        # ax.tick_params(axis="x", labelsize=self.internal_parameters["xtick_size"], labelrotation=self.internal_parameters["xrotation"])
        # ax.tick_params(axis="y", labelsize=self.internal_parameters["ytick_size"], labelrotation=self.internal_parameters["yrotation"])

        # ax.set_xlabel("Chromosomes", fontsize=self.internal_parameters["label_size"])
        # ax.set_ylabel(r"$-log_{10}(p)$", fontsize=self.internal_parameters["label_size"])

        # title: str = self.internal_parameters["title"]
        # if title:
        #     ax.set_title(title, fontsize=self.internal_parameters["title_size"])

        # # if isAx:
        # #     fig.tight_layout()

        # if self.internal_parameters["show"]:
        #     plt.show()

        # plt.savefig(os.path.join(self.output, title), format="png")

        # plt.clf()
        # plt.close()

            
#################################################################################################################
# Function that is responsible for creating the manhattan plot


# def manhattan(
#     assoc,
#     out=False,
#     gap=10,
#     ax=False,
#     cmap=False,
#     cmap_var=2,
#     col_chr="CHR",
#     col_bp="BP",
#     col_p="P",
#     col_snp="SNP",
#     show=False,
#     title=False,
#     xtick_size=10,
#     ytick_size=10,
#     xrotation=0,
#     yrotation=0,
#     label_size=15,
#     title_size=20,
#     suggestiveline=-np.log10(1e-5),
#     genomewideline=-np.log10(5e-8),
#     **kwargs,
# ):

#     # # This command is making sure that the cmap_var is either an instance of a Number or a list
#     # if not (isinstance(cmap_var, numbers.Number) or isinstance(cmap_var, list)):
#     # 	raise Exception("[ERROR]: cmap_var should either be list or number.")

#     # list_color = list()
#     # if cmap:
#     # 	try:
#     # 		if isinstance(cmap_var, numbers.Number):
#     # 			step = int(len(cmap.colors) / cmap_var)

#     # 			for i, color in enumerate(cmap.colors):
#     # 				if i % step == 0:
#     # 					list_color.append(colors.to_hex(color))
#     # 		else:
#     # 			list_color = cmap_var

#     # 	except AttributeError:
#     # 		if isinstance(cmap_var, numbers.Number):
#     # 			step = int(256 / cmap_var)

#     # 			for i in range(256):
#     # 				if i % step == 0:
#     # 					list_color.append(colors.to_hex(cmap(i)[:3]))
#     # 		else:
#     # 			list_color = cmap_var
#     # else:
#     # 	cmap = plt.get_cmap("Greys_r")
#     # 	if isinstance(cmap_var, numbers.Number):
#     # 		if cmap_var == 2:
#     # 			list_color = [colors.to_hex(cmap(0)[:3]), colors.to_hex(cmap(80)[:3])]
#     # 		else:
#     # 			step = int(256 / cmap_var)

#     # 			for i in range(256):
#     # 				if i % step == 0:
#     # 					list_color.append(colors.to_hex(cmap(i)[:3]))
#     # 	else:
#     # 		list_color = cmap_var

#     if not (ax or show or out):
#         raise Exception("[ERROR]: Either of the ax, show, and out must have a value.")
#     isAx = not ax
#     # if isinstance(assoc, str):
#     #     df_assoc = pd.read_csv(assoc, header=0, delim_whitespace=True)
#     # elif isinstance(assoc, pd.DataFrame):
#     #     df_assoc = assoc
#     # else:
#     #     raise Exception(
#     #         "[ERROR]: assoc must be either string(path) or pandas.DataFrame."
#     #     )

#     # if col_chr not in df_assoc.columns:
#     #     raise Exception("[ERROR]: Column '{0}' not found!".format(col_chr))
#     # if col_bp not in df_assoc.columns:
#     #     raise Exception("[ERROR]: Column '{0}' not found!".format(col_bp))
#     # if col_p not in df_assoc.columns:
#     #     raise Exception("[ERROR]: Column '{0}' not found!".format(col_p))
#     if col_snp not in df_assoc.columns:
#         print("[WARNING]: Column '{0}' not found!".format(col_snp))

#     # df_assoc[col_chr] = df_assoc[col_chr].astype("category")
#     # df_assoc[col_bp] = df_assoc[col_bp].astype(int)
#     # df_assoc[col_p] = df_assoc[col_p].astype(float)
#     # df_assoc = df_assoc.sort_values([col_chr, col_bp])
#     # we are appeniding the number of variants for each chromosome to this list
#     # chr_len = list()

#     # for cChr in df_assoc[col_chr].unique():
#     #     chr_len.append(len(df_assoc[df_assoc[col_chr] == cChr]))
#     # # We are then creaing a weight for the gaps by taking the min_value/100 and converting to an int
#     # weight_gap = int(min(chr_len) / 100)

#     # list_ind = list()
#     # for cChr in df_assoc[col_chr].unique():
#     #     if len(list_ind) == 0:
#     #         last_ind = 0
#     #     else:
#     #         last_ind = (list_ind[-1] + 1) + (gap * weight_gap)

#     #     list_ind += [
#     #         last_ind + num for num in range(len(df_assoc[df_assoc[col_chr] == cChr]))
#     #     ]

#     # df_assoc["IND"] = list_ind
#     # df_assoc["LOG_P"] = -np.log10(df_assoc[col_p])

#     # if isAx:
#     #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
#     # else:
#     #     fig = ax.figure

#     x_ticks, x_labels = list(), list()

#     for i, cChr in enumerate(df_assoc[col_chr].unique()):
#         ind = df_assoc[df_assoc[col_chr] == cChr]["IND"]
#         log_p = df_assoc[df_assoc[col_chr] == cChr]["LOG_P"]

#         ax.scatter(
#             ind, log_p, marker=".", s=5, color=list_color[i % cmap_var], **kwargs
#         )
#         x_ticks.append(ind.iloc[0] + (ind.iloc[-1] - ind.iloc[0]) / 2)
#         x_labels.append(cChr)

#     x_padding = len(list_ind) / 20
#     xlim_min = list_ind[0] - x_padding
#     xlim_max = list_ind[-1] + x_padding

#     if suggestiveline:
#         ax.plot([xlim_min, xlim_max], [suggestiveline, suggestiveline], "b-")
#     if genomewideline:
#         ax.plot([xlim_min, xlim_max], [genomewideline, genomewideline], "r-")
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)

#     ax.set_xticks(x_ticks)
#     ax.set_xticklabels(x_labels)
#     # ax.set_ylim(bottom=0,top=math.floor(1+max(df_assoc["LOG_P"])+(max(df_assoc["LOG_P"])/100)))
#     # ax.set_ylim(bottom=0,top=max(df_assoc["LOG_P"]))
#     ax.set_ylim(bottom=0)
#     ax.set_xlim([xlim_min, xlim_max])

#     ax.tick_params(axis="x", labelsize=xtick_size, labelrotation=xrotation)
#     ax.tick_params(axis="y", labelsize=ytick_size, labelrotation=yrotation)

#     ax.set_xlabel("Chromosomes", fontsize=label_size)
#     ax.set_ylabel(r"$-log_{10}(p)$", fontsize=label_size)

#     if title:
#         ax.set_title(title, fontsize=title_size)

#     if isAx:
#         fig.tight_layout()

#     if show:
#         plt.show()

#     if out:
#         plt.savefig(out, format="png")

#     if isAx:
#         plt.clf()
#         plt.close()


def qqplot(
    assoc,
    out=False,
    col_p="P",
    show=False,
    ax=False,
    title=False,
    xtick_size=10,
    ytick_size=10,
    xrotation=0,
    yrotation=0,
    label_size=15,
    title_size=20,
    **kwargs,
):
    if not (ax or show or out):
        raise Exception("[ERROR]: Either of the ax, show, and out must have a value.")

    isAx = not ax
    p_vals = None
    if isinstance(assoc, str):
        df_assoc = pd.read_csv(
            assoc,
            header=0,
            delim_whitespace=True,
            dtype={
                1: "category",
                2: int,
                4: float,
                5: float,
                7: float,
                8: float,
                9: float,
            },
        )
        p_vals = df_assoc[col_p].dropna()
        p_vals = p_vals[(0 < p_vals) & (p_vals < 1)]

    elif isinstance(assoc, pd.DataFrame):
        p_vals = assoc[col_p].dropna()
        p_vals = p_vals[(0 < p_vals) & (p_vals < 1)]
    elif isinstance(assoc, pd.Series):
        p_vals = assoc.dropna()
        p_vals = p_vals[(0 < p_vals) & (p_vals < 1)]
    else:
        p_vals = [ele for ele in assoc if (0 < ele) and (ele < 1)]

    observed = -np.log10(np.sort(np.array(p_vals)))
    expected = -np.log10(ppoints(len(p_vals)))

    x_padding = (np.nanmax(expected) - np.nanmin(expected)) / 12
    y_padding = (np.nanmax(observed) - np.nanmin(observed)) / 12

    if isAx:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    else:
        fig = ax.figure

    ax.scatter(expected, observed, c="k", **kwargs)

    xlim_min = np.nanmin(expected) - x_padding
    xlim_max = np.nanmax(expected) + x_padding
    ylim_min = np.nanmin(observed) - y_padding
    ylim_max = np.nanmax(observed) + y_padding

    max_lim = xlim_max if xlim_max < ylim_max else ylim_max
    min_lim = xlim_min if xlim_min > ylim_min else ylim_min
    ax.plot([min_lim, max_lim], [min_lim, max_lim], "r-")

    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([ylim_min, ylim_max])
    ax.set_xlabel("Expected $-log_{10}(p)$", fontsize=label_size)
    ax.set_ylabel("Observed $-log_{10}(p)$", fontsize=label_size)

    ax.tick_params(axis="x", labelsize=xtick_size, labelrotation=xrotation)
    ax.tick_params(axis="y", labelsize=ytick_size, labelrotation=yrotation)

    if title:
        ax.set_title(title, fontsize=title_size)

    if isAx:
        fig.tight_layout()

    if show:
        plt.show()

    if out:
        plt.savefig(out, format="png")

    if isAx:
        plt.clf()
        plt.close()


if __name__ == "__main__":
    FORMAT = "%(levelname)s %(asctime)-15s %(name)-20s %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="PGMplatform")

    parser.add_argument("--assoc", type=str, help=" ", default="./", required=True)
    parser.add_argument("--out", type=str, help=" ", default=None, required=False)
    parser.add_argument("--gap", type=int, help=" ", default=10, required=False)
    parser.add_argument("--plot", type=str, help="[manhattan/qqplot]", required=True)
    parser.add_argument("--show", action="store_true", required=False)

    args = parser.parse_args()

    logger.info(args)

    try:
        if args.plot in ["Manhattan", "manhattan"]:
            if args.out == None:
                manhattan(args.assoc, "./Manhattan.png", args.gap, show=args.show)
            else:
                manhattan(args.assoc, args.out, args.gap, show=args.show)
        elif args.plot in ["QQplot", "qqplot", "qq", "QQ", "QQPlot"]:
            if args.out == None:
                qqplot(args.assoc, "./QQplot.png")
            else:
                qqplot(args.assoc, args.out)

    except Exception:
        logger.error(traceback.format_exc())
        logger.error("qqman.py failure on arguments: {}".format(args))
