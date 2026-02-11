#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import os
import argparse
import pandas as pd
import networkx as nx
import pydot
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from IPython.display import Image, display
from gprofiler import GProfiler
import gseapy as gp
import numpy as np
import ast
from upsetplot import UpSet, from_contents
from ridgeplot import ridgeplot
from typing import List, Tuple
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
import matplotlib.cm as cm
import scipy.stats as stats
import seaborn as sns
from scipy.stats import mannwhitneyu
import warnings
import os
warnings.filterwarnings('ignore')

# Allow disabling SSL verification for GProfiler connections (useful for corporate firewalls/proxies)
# Set environment variable PYPATHOMICS_DISABLE_SSL_VERIFY=1 to disable SSL verification
_DISABLE_SSL_VERIFY = os.environ.get('PYPATHOMICS_DISABLE_SSL_VERIFY', '0') == '1'

def performFEA(
    geneList,
    organism='hsapiens',
    sources='GO:All',
    pvalue=0.05,
    noGene=False,
    noIEA=True,
    method='fdr'
):
    """
    Perform functional enrichment analysis (FEA) on a list of genes and return the enrichment results as a DataFrame.
    - This function uses the GProfiler tool to perform enrichment analysis on the provided gene list.
    - The analysis can be customized by specifying organism, source databases, significance threshold, and other parameters.
    - The results are returned as a DataFrame containing functional terms (e.g., GO terms) and their statistical significance.

    Args:
    - geneList: list of gene identifiers.
    - organism: string, specifies the organism for analysis (default is 'hsapiens' for human).
    - sources: string or list, specifies the source databases for enrichment analysis (default is 'GO:All' for all GO terms).
        Supported options include:
        - 'GO:All' for all GO terms
        - 'GO:MF' for Molecular Function
        - 'GO:CC' for Cellular Component
        - 'GO:BP' for Biological Process
        - 'KEGG' for Kyoto Encyclopedia of Genes and Genomes
        - 'REAC' for Reactome
        - 'WP' for WikiPathways
        - 'TF' for Transcription Factors (Transfac)
        - 'MIRNA' for miRNA interactions (miRTarBase)
        - 'HPA' for Human Protein Atlas
        - 'CORUM' for CORUM protein complexes
        - 'HP' for Human Phenotype Ontology
    - pvalue: float, user-defined significance threshold (default is 0.05).
    - noGene: bool, if True, the 'intersection' column (genes related to the query) is not returned (default is False).
    - noIEA: bool, if True, excludes annotations inferred from electronic annotation (IEA) without experimental support (default is True).
    - method: string, specifies the method for p-value correction (default is 'fdr'). Supported methods:
        - 'g_SCS': Graph-based Sequential Correction Strategy
        - 'bonferroni': Bonferroni correction
        - 'fdr': False Discovery Rate

    Returns:
    - DataFrame containing the enrichment analysis results, including term IDs, names, significance levels, adjusted p-values, etc.
    """
    if sources == 'GO:All':
        sources = ['GO:BP', 'GO:CC', 'GO:MF']
    
    # Configure SSL settings if needed
    if _DISABLE_SSL_VERIFY:
        print("Warning: SSL verification is disabled (PYPATHOMICS_DISABLE_SSL_VERIFY=1)")
        import ssl
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Monkey-patch ssl to disable verification globally (required for some network environments)
        ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        gp = GProfiler(return_dataframe=True)
        print(f"Connecting to GProfiler service for enrichment analysis...")
        
        enrichDf = gp.profile(
            query=geneList,
            organism=organism,
            sources=sources,
            user_threshold=pvalue,
            no_evidences=noGene,
            no_iea=noIEA,
            significance_threshold_method=method
        )
        
        columns_to_drop = ['effective_domain_size', 'query', 'evidences']
        enrichDf = enrichDf.drop(columns_to_drop, axis=1)
        print(f"GProfiler enrichment analysis completed successfully.")
        return enrichDf
        
    except Exception as e:
        error_msg = str(e)
        print(f"GProfiler connection failed: {error_msg}")
        
        # Provide helpful troubleshooting advice
        if 'SSL' in error_msg or 'certificate' in error_msg.lower():
            hint = (
                "\n"
                "SSL/Certificate error detected. Possible solutions:\n"
                "1. Set environment variable: PYPATHOMICS_DISABLE_SSL_VERIFY=1\n"
                "2. Check your network proxy settings\n"
                "3. Ensure firewall allows connections to biit.cs.ut.ee\n"
                "4. Try running from a different network environment\n"
            )
            print(hint)
            raise RuntimeError(f"GProfiler SSL connection error: {error_msg}{hint}")
        else:
            raise RuntimeError(f"GProfiler connection failed: {error_msg}")

def performDAG(obofilepath, goTerms, highlightColor, fileName):
    """
    Generate a Directed Acyclic Graph (DAG) to display given GO terms and their relationships, and save it as a PNG file.

    Parameters:
    - obofilepath: string, path to the OBO file.
    - goTerms: list of GO term IDs to display.
    - highlightColor: string, color code to highlight specific nodes.
    - fileName: string, name of the output file (PNG format).
    """

    filePath = obofilepath
    terms = parseOboFile(filePath)
    G = buildGraph(terms)

    # Format goTerms list, ensuring each term is enclosed in double quotes
    formattedGoTerms = [f'"{term}"' if not term.startswith('"') else term for term in goTerms]

    subgraphs = [getAncestorsAndDescendants(G, term) for term in formattedGoTerms]
    mergedSubgraph = mergeSubgraphs(subgraphs)
    P_example = modifiedToPydot(mergedSubgraph, formattedGoTerms, highlightColor)

    # Add graph title
    title = "GO Term Hierarchy: Parent and Child Relationships"
    P_example.set_label(title)
    P_example.set_labelloc('t')  # 't' means top

    P_example.set_rankdir('TB')  # 'TB' for top-bottom layout
    png_str_example = P_example.create_png(prog='dot')
    display(Image(data=png_str_example))

    # Save the graph as a PNG file
    P_example.write_png(fileName)

def parseOboFile(filePath):
    """
    Parse an OBO format file to extract term information.

    Parameters:
    - filePath: string, path to the OBO file.

    Yields:
    - Generator that yields term information as dictionaries, including id, name, and is_a (parent terms).
    """
    currentTerm = None
    with open(filePath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[Term]'):
                if currentTerm:
                    yield currentTerm
                currentTerm = {}
            elif line.startswith('id:'):
                currentTerm['id'] = line.split('id: ')[1]
            elif line.startswith('name:'):
                currentTerm['name'] = line.split('name: ')[1]
            elif line.startswith('is_a:'):
                parentId = line.split('is_a: ')[1].split(' ! ')[0]
                if 'is_a' not in currentTerm:
                    currentTerm['is_a'] = []
                currentTerm['is_a'].append(parentId)
    if currentTerm:
        yield currentTerm

def buildGraph(terms):
    """
    Build a directed graph based on terms from an OBO file.

    Parameters:
    - terms: iterator containing term information parsed from the OBO file.

    Returns:
    - NetworkX DiGraph representing relationships between terms.
    """
    G = nx.DiGraph()
    for term in terms:
        nodeName = f'"{term["id"]}"'
        G.add_node(nodeName, label=term['name'])
        for parentId in term.get('is_a', []):
            G.add_edge(f'"{parentId}"', nodeName)
    return G

def getAncestorsAndDescendants(graph, term):
    """
    Retrieve all ancestors and descendants of a specified term.

    Parameters:
    - graph: DiGraph containing term relationships.
    - term: string, the term ID to find relatives for.

    Returns:
    - Subgraph containing the specified term and all its ancestors and descendants.
    """
    ancestors = nx.ancestors(graph, term)
    descendants = nx.descendants(graph, term)
    related_terms = set(ancestors | descendants | {term})
    return graph.subgraph(related_terms).copy()

def mergeSubgraphs(graphs):
    """
    Merge multiple subgraphs into a single directed graph.

    Parameters:
    - graphs: list of DiGraph objects.

    Returns:
    - Merged DiGraph object.
    """
    mergedGraph = nx.DiGraph()
    for g in graphs:
        mergedGraph = nx.compose(mergedGraph, g)
    return mergedGraph

def modifiedToPydot(G, highlightNodes, highlightColor):
    """
    Convert a NetworkX graph to a Pydot graph and highlight specific nodes.

    Parameters:
    - G: DiGraph object containing term relationships.
    - highlightNodes: list of node IDs to highlight.
    - highlightColor: string, color code for highlighted nodes.

    Returns:
    - Pydot Dot object representing the graph with highlighted nodes.
    """
    P = pydot.Dot(graph_type='digraph', splines='ortho')
    for n in G.nodes():
        go_id = n.replace('"', '')
        termName = G.nodes[n].get('label', '')
        label = f'{go_id}\n{termName}'
        if n in highlightNodes:
            node = pydot.Node(n,
                              label=label,
                              shape='ellipse',
                              style="filled",
                              fillcolor=highlightColor)
        else:
            node = pydot.Node(n, label=label, shape='rectangle')
        P.add_node(node)
    for u, v in G.edges():
        edge = pydot.Edge(u, v, style='solid')
        P.add_edge(edge)
    return P

def performCreateBarplot(df,
                         xData,
                         yData,
                         numTop,
                         colorBy,
                         colors,
                         title=None,
                         facetGrid=None,
                         xAxisTitle=None,
                         yAxisTitle=None,
                         width=None,
                         height=None,
                         save_dir=None):
    """
    Create and display a bar plot to showcase the top N entries in the dataset.

    Parameters:
    - df: DataFrame containing the data for plotting.
    - xData: string, column name for the x-axis data.
    - yData: string, column name for the y-axis data.
    - numTop: int, number of top entries to display per group.
    - colorBy: string, column name to color the bars.
    - colors: list of color codes for the bars.
    - title: string, title of the bar plot.
    - facetGrid: string (optional), column name to create a facet grid.
    - xAxisTitle: string (optional), title for the x-axis.
    - yAxisTitle: string (optional), title for the y-axis.
    - width: int (optional), width of the figure.
    - height: int (optional), height of the figure.
    - save_dir: string, directory to save the figure.
    """

    topPlot = df.groupby('source').head(numTop).reset_index(drop=True)

    if facetGrid:
        fig = px.bar(topPlot,
                     x=xData,
                     y=yData,
                     color=colorBy,
                     facet_row=facetGrid,
                     color_discrete_sequence=colors,
                     orientation='h')
    else:
        fig = px.bar(topPlot,
                     x=xData,
                     y=yData,
                     color=colorBy,
                     color_discrete_sequence=colors,
                     orientation='h')

    # Remove original axis titles and update layout
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text=None)
    if facetGrid:
        fig.update_yaxes(matches=None)

    bottomMargin = 100 if xAxisTitle else 50
    leftMargin = 150 if yAxisTitle else 100
    width = width if width else 1200
    height = height if height else 100 * numTop

    fig.update_layout(title={
        'text': title,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
                      width=width,
                      height=height,
                      margin=dict(l=leftMargin, b=bottomMargin))

    # Add x-axis and y-axis titles
    if xAxisTitle:
        fig.add_annotation(text=xAxisTitle,
                           xref="paper",
                           yref="paper",
                           x=0.5,
                           y=-0.15,
                           showarrow=False,
                           font=dict(size=16))
    if yAxisTitle:
        fig.add_annotation(text=yAxisTitle,
                           xref="paper",
                           yref="paper",
                           x=-0.15,
                           y=0.5,
                           xanchor='right',
                           showarrow=False,
                           textangle=-90,
                           font=dict(size=16))

    # Display and save the figure
    # fig.show()
    if save_dir != None:
        fig.write_image(os.path.join(save_dir, "BarPlot.png"), format="png")

def termsGeneNet(enrichmentDf,
                 termColumn='native',
                 geneColumn='intersections',
                 layoutMethod='kamada',
                 edgeColor='#888',
                 edgeWidth=0.5,
                 termColor='blue',
                 termSize=20,
                 sharedGeneColor='red',
                 sharedGeneSize=15,
                 uniqueGeneColor='green',
                 uniqueGeneSize=10,
                 nodeOutlineColor='white',
                 nodeOutlineWidth=2,
                 labelTypes=[],
                 title=None,
                 width=1200,
                 height=1000,
                 save_dir=None):
    """
    Plot a term-gene network graph.

    Parameters:
    - enrichmentDf: DataFrame containing enrichment analysis results.
    - termColumn: string, column name containing terms (default 'native').
    - geneColumn: string, column name containing gene lists (default 'intersections').
    - layoutMethod: string, layout method ('kamada', 'spring', 'circular').
    - edgeColor: string, color of the edges.
    - edgeWidth: float, width of the edges.
    - termColor: string, color of term nodes.
    - termSize: int, size of term nodes.
    - sharedGeneColor: string, color of shared gene nodes.
    - sharedGeneSize: int, size of shared gene nodes.
    - uniqueGeneColor: string, color of unique gene nodes.
    - uniqueGeneSize: int, size of unique gene nodes.
    - nodeOutlineColor: string, color of node outlines.
    - nodeOutlineWidth: int, width of node outlines.
    - labelTypes: list, types of nodes to label ('terms', 'shared_genes', 'unique_genes').
    - title: string, title of the graph.
    - width: int, width of the figure.
    - height: int, height of the figure.
    - save_dir: string, directory to save the figure.
    """

    # Split gene IDs and create edges
    edges = []
    for _, row in enrichmentDf.iterrows():
        if isinstance(row[geneColumn], str):
            genes = eval(row[geneColumn])
        else:
            genes = row[geneColumn]
        for gene in genes:
            edges.append((row[termColumn], gene))

    # Create graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Distinguish node types and identify shared genes
    terms = set([edge[0] for edge in edges])
    genes = set([edge[1] for edge in edges])
    shared_genes = {
        gene
        for gene in genes if sum(1 for edge in edges if edge[1] == gene) > 1
    }

    # Get node positions
    if layoutMethod == 'kamada':
        pos = nx.kamada_kawai_layout(G)
    elif layoutMethod == 'spring':
        pos = nx.spring_layout(G)
    elif layoutMethod == 'circular':
        pos = nx.circular_layout(G)

    # Create edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x,
                            y=edge_y,
                            line=dict(width=edgeWidth, color=edgeColor),
                            hoverinfo='none',
                            mode='lines',
                            name='Edges')

    # Initialize node trace list
    nodeTraces = []

    # Create traces for each node type
    nodeTypeLabels = ['Terms', 'Shared Genes', 'Unique Genes']
    for (nodeType, color, size, includeLabel), label in zip(
        [(terms, termColor, termSize, 'terms' in labelTypes),
         (shared_genes, sharedGeneColor, sharedGeneSize, 'shared_genes' in labelTypes),
         (genes - shared_genes, uniqueGeneColor, uniqueGeneSize, 'unique_genes' in labelTypes)], nodeTypeLabels):
        x, y, text = [], [], []
        for node in nodeType:
            x.append(pos[node][0])
            y.append(pos[node][1])
            text.append(node if includeLabel else '')
        nodeTraces.append(
            go.Scatter(
                x=x,
                y=y,
                mode='markers+text' if includeLabel else 'markers',
                text=text,
                hoverinfo='text',
                marker=dict(color=color,
                            size=size,
                            line=dict(color=nodeOutlineColor,
                                      width=nodeOutlineWidth)),
                name=label))

    # Combine traces and create figure
    fig = go.Figure(data=[edge_trace] + nodeTraces)
    fig.update_layout(title={
        'text': title,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 35}
    },
                      showlegend=True,
                      hovermode='closest',
                      xaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showticklabels=False),
                      yaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showticklabels=False),
                      width=width,
                      height=height,
                      legend=dict(x=1, y=1, xanchor='right', yanchor='top', bgcolor='lightblue'),
                      plot_bgcolor='lightblue')

    # Display and save the figure
    # fig.show()
    if save_dir!=None:
        fig.write_image(os.path.join(save_dir, "TermGeneNet.png"), format="png")

def plotUpSet(enrichmentDf,
              termColumn='native',
              geneColumn='intersections',
              width=None,
              height=None,
              save_dir=None):
    """
    Plot an UpSetPlot to visualize the intersections of gene sets.

    Parameters:
    - enrichmentDf: DataFrame containing enrichment results.
    - termColumn: string, column name containing terms (default 'native').
    - geneColumn: string, column name containing gene lists (default 'intersections').
    - width: float, width of the figure in inches.
    - height: float, height of the figure in inches.
    - save_dir: string, directory to save the figure.
    """

    # Convert string representations of lists to actual lists
    if isinstance(enrichmentDf[geneColumn], str):
        enrichmentDf[geneColumn] = enrichmentDf[geneColumn].apply(ast.literal_eval)

    # Convert to UpSetPlot format
    contents = {
        row[termColumn]: set(row[geneColumn])
        for index, row in enrichmentDf.iterrows()
    }
    upset_data = from_contents(contents)

    # Create UpSet plot
    upset = UpSet(upset_data, subset_size='count', show_counts=True)

    # Plot
    upset.plot()

    # Set figure size if provided
    if width is not None or height is not None:
        plt.gcf().set_size_inches(
            width if width is not None else plt.gcf().get_size_inches()[0],
            height if height is not None else plt.gcf().get_size_inches()[1])
    plt.suptitle('Unique Gene Counts in GO Terms', x=0.5, fontsize=15)

    # Save and display the figure
    if save_dir != None:
        plt.savefig(os.path.join(save_dir, 'upsetplot.png'))
    # plt.show()

def createWideFormatDf(enrichmentDf,
                       geneListDf,
                       termColumn='native',
                       geneColumn='intersections',
                       drawColumn='LogFC',
                       fillNA='NA'):
    """
    Convert enrichment analysis results and gene expression information into a wide-format DataFrame.

    Parameters:
    - enrichmentDf: DataFrame containing GO enrichment results.
    - geneListDf: DataFrame containing gene identifiers and corresponding values to plot.
    - termColumn: string, column name for GO terms (default 'native').
    - geneColumn: string, column name for gene lists (default 'intersections').
    - drawColumn: string, column name for the values to draw (default 'logFC').
    - fillNA: value to replace missing values with (default 'NA').
    """

    long_format_data = []

    for _, row in enrichmentDf.iterrows():
        category = row[termColumn]
        if isinstance(row[geneColumn], str):
            intersections = row[geneColumn][1:-1].replace("'", "").split(', ')
        else:
            intersections = row[geneColumn]

        for gene in intersections:
            gene_row = geneListDf[geneListDf['Gene ID'] == gene]
            if not gene_row.empty:
                value = gene_row[drawColumn].values[0]
                long_format_data.append({
                    'category': category,
                    'gene': gene,
                    'value': value
                })

    long_format_df = pd.DataFrame(long_format_data)

    wideFormatDf = long_format_df.pivot_table(index='gene',
                                              columns='category',
                                              values='value',
                                              aggfunc=np.sum)

    if not fillNA:
        wideFormatDf = wideFormatDf.fillna(fillNA)

    return wideFormatDf

def generateRidgePlot(wideFormatDf,
                      colorscale="viridis",
                      colormode="row-index",
                      coloralpha=0.65,
                      labels=None,
                      linewidth=2,
                      spacing=9 / 9,
                      height=None,
                      width=None,
                      fontSize=12,
                      plotBgcolor="white",
                      title="Distribution of LogFC Values per GO Term",
                      xAxisTitle="LogFC",
                      yAxisTitle="GO Terms",
                      showLegend=False,
                      save_dir=None):
    """
    Generate and display a ridge plot using the ridgeplot library.

    Parameters:
    - wideFormatDf: DataFrame in wide format, each column is a distribution.
    - colorscale: string, name of the color scale.
    - colormode: string, color mode.
    - coloralpha: float, transparency of colors.
    - labels: list of labels for each distribution.
    - linewidth: float, width of the lines.
    - spacing: float, spacing between curves.
    - title: string, title of the plot.
    - height: int, height of the figure.
    - width: int, width of the figure.
    - fontSize: int, font size.
    - plotBgcolor: string, background color.
    - xAxisTitle: string, x-axis title.
    - yAxisTitle: string, y-axis title.
    - showLegend: bool, whether to show the legend.
    - save_dir: string, directory to save the figure.
    """

    samples = wideFormatDf.to_numpy()
    labels = labels if labels is not None else wideFormatDf.columns.tolist()

    fig = ridgeplot(
        samples=samples.T,
        kde_points=np.linspace(samples.min(), samples.max(), 1000),
        colorscale=colorscale,
        colormode=colormode,
        coloralpha=coloralpha,
        labels=labels,
        linewidth=linewidth,
        spacing=spacing,
    )

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16}
        },
        height=height,
        width=width,
        font_size=fontSize,
        plot_bgcolor=plotBgcolor,
        xaxis_title=xAxisTitle,
        yaxis_title=yAxisTitle,
        xaxis_tickvals=np.linspace(samples.min(), samples.max(), num=11),
        xaxis_ticktext=[
            str(round(i, 2))
            for i in np.linspace(samples.min(), samples.max(), num=11)
        ],
        xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        showlegend=showLegend,
    )

    # Display and save the figure
    # fig.show()
    if save_dir != None:
        fig.write_image(os.path.join(save_dir, "RidgePlot.png"), format="png")

def performGSEA(sortedGeneDf,
                geneSets='MSigDB_Hallmark_2020',                     # 'MSigDB_Hallmark_2020'
                processes=1,
                permutationNum=100,
                minSize=15,
                maxSize=500,
                outdir='./GSEA(PyPathomics_fpkm)',
                seed=42):
    """
    Perform gene set enrichment analysis (GSEA) using the prerank method from gseapy.

    Args:
        - sortedGeneDf: DataFrame, sorted list of genes.
        - geneSets: string, background gene sets/database for GSEA analysis.
        - processes: int, number of processes for parallel computation.
        - permutationNum: int, number of permutations for significance testing.
        - minSize: int, minimum size of gene sets considered.
        - maxSize: int, maximum size of gene sets considered.
        - outdir: string, output directory to save GSEA results.
        - seed: int, random seed for reproducibility.

    Returns:
        gseapy.gsea.GSEA object including attributes
        -res2d (Pandas.DataFrame,the core result data, including enrichment statistics for all gene sets (such as p-values, etc.).)
        -gene_sets (gene set dictionary for analysis)
        -ranking (pandas.Series, ranked genes and corresponding logFC (log Fold Change) values)
    """

    preRES = gp.prerank(rnk=sortedGeneDf,
                        gene_sets=geneSets,
                        threads=processes,
                        permutation_num=permutationNum,
                        min_size=minSize,
                        max_size=maxSize,
                        outdir=outdir,
                        seed=seed)

    print(preRES.res2d.head())
    if outdir:
        print(f"GSEA analysis results have been saved to the '{outdir}' folder.")

    return preRES

class GSEAPlotly(object):
    def __init__(self,
                 preRES: object,
                 tag: List[int],
                 runes: List[float],
                 nes: float,
                 pval: float,
                 fdr: float,
                 rankMetric: List[float] = None,
                 ESColor: str = "#88C544",
                 hitColor: str = "#ff9f43",
                 phenoColor: List[str] = ["#ee5253", "white", "#01a3a4"],
                 rankColor: str = "#8ECFC9",
                 bgColor: str = "white",
                 gridColor: str = "rgba(0, 0, 0, 0.1)",
                 gridDash: str = "dot",
                 borderColor: str = "rgba(0, 0, 0, 0.1)",
                 zeroColor: str = "#9980FA",
                 zeroDash: str = "dash",
                 title: str = None,
                 figSize: Tuple[int, int] = (800, 700),
                 fileName: str = None):
        """
        Initialize an instance of the GSEAPlotly class.

        Parameters:
        - preRES: GSEA results object.
        - tag: List[int], indices of hits in gene set S.
        - runes: List[float], running enrichment scores.
        - nes: float, normalized enrichment score.
        - pval: float, nominal P-value.
        - fdr: float, false discovery rate.
        - rankMetric: List[float], ranked metric values.
        - ESColor: str, color of the enrichment score plot.
        - hitColor: str, color for hits.
        - phenoColor: List[str], colors for phenotype gradient.
        - rankColor: str, color of the ranked metric plot.
        - bgColor: str, background color.
        - gridColor: str, color of grid lines.
        - gridDash: str, style of grid lines.
        - borderColor: str, color of borders.
        - zeroColor: str, color of the zero score line.
        - zeroDash: str, style of the zero score line.
        - title: str, title of the plot.
        - figSize: Tuple[int, int], size of the figure (width, height).
        - fileName: str, name of the output file.
        """
        self.preRES = preRES
        self.tag = tag
        self.runes = runes
        self.nes = nes
        self.pval = pval
        self.fdr = fdr
        self.rankMetric = rankMetric if rankMetric is not None else []
        self.ESColor = ESColor
        self.hitColor = hitColor
        self.phenoColor = phenoColor
        self.rankColor = rankColor
        self.bgColor = bgColor
        self.gridColor = gridColor
        self.gridDash = gridDash
        self.borderColor = borderColor
        self.zeroColor = zeroColor
        self.zeroDash = zeroDash
        self.title = title
        self.figSize = figSize
        self.fileName = fileName

        # Capitalize title appropriately
        go_index = title.find("GO")
        if go_index != -1:
            before_go = title[:go_index]
            after_go = title[go_index:]
            formatted_before_go = before_go.capitalize()
            self.title = formatted_before_go + after_go
        else:
            self.title = title.capitalize()

        self.fig = make_subplots(rows=4,
                                 cols=1,
                                 shared_xaxes=True,
                                 vertical_spacing=0,
                                 row_heights=[0.25, 0.05, 0.05, 0.65]
                                 if rankMetric is not None else [0, 0, 0, 1])
        self.fig.update_layout(height=self.figSize[1], width=self.figSize[0])

    def plot_rank_metric(self):
        if not self.rankMetric.empty:
            self.fig.add_trace(go.Scatter(x=list(range(len(self.rankMetric))),
                                          y=self.rankMetric,
                                          fill='tozeroy',
                                          line=dict(color=self.rankColor),
                                          name='Ranking Metric Scores'),
                               row=4,
                               col=1)
        self.fig.update_xaxes(title_text='Rank in Ordered Dataset',
                              row=4,
                              col=1)
        self.fig.update_yaxes(title_text='Ranked List Metric', row=4, col=1)

    def plot_gradient_bar(self):
        z = [self.rankMetric.to_list()]

        min_val = self.rankMetric.min()
        max_val = self.rankMetric.max()
        zero_pos_relative = (0 - min_val) / (max_val - min_val)

        colorscale = [
            [0.0, self.phenoColor[0]],
            [zero_pos_relative, self.phenoColor[1]],
            [1.0, self.phenoColor[2]]
        ]

        self.fig.add_trace(
            go.Heatmap(
                z=z,
                x=list(range(len(self.rankMetric))),
                y=[''],
                colorscale=colorscale,
                showscale=False,
            ),
            row=3,
            col=1)

    def plot_hits(self):
        self.fig.update_yaxes(showticklabels=False, row=2, col=1)
        max_x = len(self.rankMetric) - 1

        for tag_value in self.tag:
            self.fig.add_shape(
                type="line",
                x0=tag_value,
                y0=0,
                x1=tag_value,
                y1=1,
                line=dict(color=self.hitColor, width=3),
                xref="x",
                yref="y domain",
                row=2,
                col=1)
            self.fig.add_trace(
                go.Scatter(
                    x=[tag_value],
                    y=[1],
                    mode='markers',
                    marker=dict(color='rgba(0,0,0,0)'),
                    hoverinfo='x',
                    hoverlabel=dict(namelength=0),
                    showlegend=False),
                row=2,
                col=1)

        self.fig.update_xaxes(range=[0, max_x], row=2, col=1)

        self.fig.add_trace(go.Scatter(x=[None],
                                      y=[None],
                                      mode='lines',
                                      line=dict(color=self.hitColor, width=3),
                                      showlegend=True,
                                      name='Hits'),
                           row=2,
                           col=1)

    def plot_enrichment_score(self):
        self.fig.add_trace(go.Scatter(x=list(range(len(self.runes))),
                                      y=self.runes,
                                      line=dict(color=self.ESColor),
                                      name='Enrichment Profile'),
                           row=1,
                           col=1)

        min_y = min(self.runes)
        annotation_text = f"NES: {self.nes:.3f}<br>P-value: {self.pval:.3e}<br>FDR: {self.fdr:.3e}"
        self.fig.add_annotation(
            xref="x1",
            yref="y1",
            x=0,
            y=min_y,
            xanchor='left',
            yanchor='bottom',
            showarrow=False,
            text=annotation_text,
            align='left')

        self.fig.update_yaxes(title_text='Enrichment Score', row=1, col=1)

    def plot_zero_score(self):
        zero_score_index = self.rankMetric.sub(0).abs().idxmin()
        zero_score_position = np.where(self.preRES.ranking.index == zero_score_index)[0][0]

        self.fig.add_shape(
            x0=zero_score_position,
            y0=0,
            x1=zero_score_position,
            y1=1,
            line=dict(
                color=self.zeroColor,
                width=2,
                dash=self.zeroDash,
            ),
            xref="x",
            yref="paper"
        )

        self.fig.add_annotation(
            x=zero_score_position,
            y=0,
            text=f"Zero score at {zero_score_position}",
            showarrow=False,
            xref="x",
            yref="y4",
            xanchor="center",
        )

    def add_border_lines(self):
        for i in range(1, 5):
            if i in [2, 3]:
                self.fig.update_xaxes(
                    showline=False,
                    showgrid=False,
                    row=i,
                    col=1)
                self.fig.update_yaxes(
                    showline=True,
                    linewidth=1.5,
                    linecolor=self.borderColor,
                    mirror=True,
                    showgrid=False,
                    row=i,
                    col=1)
            else:
                self.fig.update_xaxes(
                    showline=True,
                    linewidth=1.5,
                    linecolor=self.borderColor,
                    mirror=True,
                    gridcolor=self.gridColor,
                    griddash=self.gridDash,
                    row=i,
                    col=1)
                self.fig.update_yaxes(
                    showline=True,
                    linewidth=1.5,
                    linecolor=self.borderColor,
                    mirror=True,
                    gridcolor=self.gridColor,
                    griddash=self.gridDash,
                    row=i,
                    col=1)

        self.fig.update_layout(
            title={
                'text': self.title,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            plot_bgcolor=self.bgColor,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5))

    def savefig(self):
        if self.fileName:
            pio.write_image(self.fig, self.fileName, format="png")

    def show(self):
        self.plot_rank_metric()
        self.plot_gradient_bar()
        self.plot_hits()
        self.plot_enrichment_score()
        self.plot_zero_score()
        self.add_border_lines()
        # self.fig.show()

def generateHeatmap(wideFormatDf,
                    colorscale='viridis',
                    colorbarTitle='LogFC',
                    title=None,
                    height=None,
                    width=None,
                    fontSize=18,
                    plotBgcolor='white',
                    xAxisTitle='Genes',
                    yAxisTitle='Terms',
                    showLegend=True,
                    save_dir=None):
    """
    Generate and display a heatmap using Plotly.

    Parameters:
    - wideFormatDf: DataFrame in wide format containing heatmap data.
    - colorscale: string, name of the color scale.
    - colorbarTitle: string, title of the color bar.
    - title: string, title of the heatmap.
    - height: int, height of the figure.
    - width: int, width of the figure.
    - fontSize: int, font size.
    - plotBgcolor: string, background color.
    - xAxisTitle: string, x-axis title.
    - yAxisTitle: string, y-axis title.
    - showLegend: bool, whether to show the legend.
    - save_dir: string, directory to save the figure.
    """

    heatmapData = wideFormatDf.values.T
    genes = wideFormatDf.index.tolist()
    goTerms = wideFormatDf.columns.tolist()

    if not width:
        width = 1000 + 15 * len(genes)
    if not height:
        height = 400 + 3 * len(goTerms)

    fig = go.Figure(data=go.Heatmap(z=heatmapData,
                                    x=genes,
                                    y=goTerms,
                                    colorscale=colorscale,
                                    colorbar=dict(title=colorbarTitle)))

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(title=xAxisTitle),
        yaxis=dict(title=yAxisTitle, autorange='reversed'),
        width=width,
        height=height,
        font_size=fontSize,
        plot_bgcolor=plotBgcolor,
        xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        showlegend=showLegend)

    # Display and save the figure
    # fig.show()
    if save_dir !=None:
        fig.write_image(os.path.join(save_dir, "HeatPlot.png"), format="png")

def generate_clinical_phenotype(gene_list_dir, feature_id, clinical_information, clinical_col, save_dir, category=True):
    clinical_suffix = clinical_information.suffix
    if clinical_suffix == '.csv':
        clinical_information = pd.read_csv(clinical_information)
    elif clinical_suffix in ['.xlsx', '.xls']:
        clinical_information = pd.read_excel(clinical_information)
    else:
        raise ValueError(f"Supported input formats of clinical information file: .csv, .xlsx, .xls")
    required_columns = ['wsi_id', 'label', clinical_col]
    for col in required_columns:
        if col not in clinical_information.columns.to_list():
            raise ValueError(f'Please ensure the following columns exist: "wsi_id", "label", "{clinical_col}".')
    clinical_information = clinical_information[required_columns]

    ### Interpolation requires all values to be numeric
    wsi_id = clinical_information[["wsi_id"]]
    clinical_information = clinical_information.drop(columns=["wsi_id"])
    non_numeric_cols = clinical_information.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"{non_numeric_cols} is/are not numeric, only numeric values and null values are allowed,"
                         f"and null values will be imputed.")
    # unique values for category before interpolation
    unique_vals = sorted(clinical_information[clinical_col].squeeze().dropna().unique())          # [1,2,3,4]
    if len(unique_vals) > 8:
        category = False

    # interpolation
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20,
        random_state=42)
    clin_interpo = imputer.fit_transform(clinical_information)
    clin_interpo = pd.DataFrame(clin_interpo, columns=clinical_information.columns)
    clin_interpo = pd.concat([wsi_id, clin_interpo], axis=1)

    if category:
        min_val = min(unique_vals)
        max_val = max(unique_vals)
        rounded_values = clin_interpo[clinical_col].round()
        clipped_values = np.clip(rounded_values, min_val, max_val)
        clipped_values = clipped_values.astype('Int64')
        clin_interpo[clinical_col] = clipped_values

    ### extraction of deg results
    # gene_list_dir 实际上传入的是 ".../<feature_id>/result/significant_genes.xlsx"
    # 用 Path 解析路径，避免 Windows 下 '/' vs '\' 导致 split 失效
    base_dir = Path(gene_list_dir).resolve().parent.parent
    long_term_feature_matrix = pd.read_csv(base_dir / "long_term" / f"long_term_{feature_id}.csv")
    short_term_feature_matrix = pd.read_csv(base_dir / "short_term" / f"short_term_{feature_id}.csv")
    # group label
    long_term_feature_matrix["group"] = "Long_term"
    short_term_feature_matrix["group"] = "Short_term"

    group_feature_matrix = pd.concat([long_term_feature_matrix, short_term_feature_matrix], axis=0)
    clin_interpo = clin_interpo[["wsi_id", clinical_col]]
    group_feature_matrix = pd.merge(group_feature_matrix, clin_interpo, on="wsi_id", how="inner")

    ### visualization
    # ratio
    if category:              # bar plot
        prop_table = pd.crosstab(group_feature_matrix["group"], group_feature_matrix[clinical_col], normalize="index")
        # chi square test
        chi2, p, dof, expected = stats.chi2_contingency(prop_table)
        n_colors = len(prop_table.columns)
        cmap = cm.get_cmap("tab20", n_colors)
        colors = [cmap(i) for i in range(n_colors)]
        ax = prop_table.plot(
            kind="bar",
            stacked=True,
            width=0.6,
            color=[colors[i] for i in range(n_colors)],
            figsize=(6.5, 6))
        # text ratio
        for idx, row in prop_table.iterrows():
            cumulative = 0
            for grade, proportion in row.items():
                if proportion > 0:
                    plt.text(
                        x=list(prop_table.index).index(idx),
                        y=cumulative + proportion / 2,
                        s=f"{proportion * 100:.1f}%",
                        ha="center", va="center", color="white", fontsize=10, fontweight="bold"
                    )
                cumulative += proportion
        if p < 0.001:
            plt.text(
                x=0.5, y=prop_table.values.max() * 1.25,
                s=f"p < 0.001",
                ha="center", va="bottom", fontsize=12, color="black"
            )
        else:
            plt.text(
                x=0.5, y=prop_table.values.max() * 1.25,
                s=f"p = {p:.4f}",
                ha="center", va="bottom", fontsize=12, color="black"
            )

        plt.ylabel(clinical_col)
        # plt.xlabel("Group")
        # plt.title("Distribution of Histologic Grade in Long vs Short groups")
        plt.legend(loc="upper right", fontsize=8)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{feature_id}/{clinical_col}_difference.png")

        ## txt report prompt
        report_lines = []
        report_lines.append(f"Analysis of differences in {clinical_col} between long and short-term groups")
        report_lines.append(f"=" * 60)
        report_lines.append(f"Clinical feature: {clinical_col}")
        report_lines.append(f"Feature type: {'categorical' if category else 'continuous'}")
        report_lines.append("")
        report_lines.append(prop_table.to_string())
        report_lines.append("")
        report_lines.append("Chi-square test results:")
        report_lines.append(f"Chi-square statistic: {chi2:.4f}")
        report_lines.append(f"p-value: {p:.4f}")
        report_lines.append(f"Degrees of freedom: {dof}")
        with open(f"{save_dir}/{feature_id}/{clinical_col}_statistics.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

    else:    # box plot
        plt.figure(figsize=(6.5, 6))
        sns.boxplot(x="group", y=clinical_col, data=group_feature_matrix, palette="Set2")
        sns.stripplot(x="group", y=clinical_col, data=group_feature_matrix, color="black", alpha=0.6, jitter=True)
        plt.ylabel(clinical_col)
        # plt.title("Distribution of Histologic Grade between Long and Short groups")

        # p value
        stat, p_value = mannwhitneyu(
            group_feature_matrix[group_feature_matrix["group"] == "Long_term"][clinical_col],
            group_feature_matrix[group_feature_matrix["group"] == "Short_term"][clinical_col],
            alternative="two-sided")
        if p_value < 0.0001:
            plt.text(0.5, max(group_feature_matrix[clinical_col]), f"p < 0.0001",
                     ha="center", fontsize=12)
        else:
            plt.text(0.5, max(group_feature_matrix[clinical_col]), f"p = {p_value:.4f}",
                     ha="center", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{feature_id}/{clinical_col}_difference.png")

        ## txt report prompt
        desc_stats = group_feature_matrix.groupby("group")[clinical_col].describe()
        report_lines = []
        report_lines.append(f"Analysis of differences in {clinical_col} between long and short-term groups")
        report_lines.append(f"=" * 60)
        report_lines.append(f"Clinical feature: {clinical_col}")
        report_lines.append(f"Feature type: {'categorical' if category else 'continuous'}")
        report_lines.append("")
        report_lines.append(desc_stats.round(3).to_string())
        report_lines.append("")
        report_lines.append("Mann-Whitney U test results:")
        report_lines.append(f"U statistic: {stat:.4f}")
        report_lines.append(f"p-value: {p_value:.4f}")
        with open(f"{save_dir}/{feature_id}/{clinical_col}_statistics.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))


def enrichment_analysis(deg_res, save_dir, sources, clinical_information, clinical_col, category=True):
    """
    :param deg_res: str, the directory of differentially expressed gene result
    :param save_dir: str, the directory of saving results
    :param sources: str or list, specifies the source databases for enrichment analysis (default is 'GO:All' for all GO terms).
    :param clinical_information: str, the directory of clinical information
    :param clinical_col: str, the column name of clinical information
    :param category: bool, the clinical information column is categorical variables or continuous variables?
        Supported options include:
        - 'GO:All' for all GO terms
        - 'GO:MF' for Molecular Function
        - 'GO:CC' for Cellular Component
        - 'GO:BP' for Biological Process
        - 'KEGG' for Kyoto Encyclopedia of Genes and Genomes
        - 'REAC' for Reactome
        - 'WP' for WikiPathways
        - 'TF' for Transcription Factors (Transfac)
        - 'MIRNA' for miRNA interactions (miRTarBase)
        - 'HPA' for Human Protein Atlas
        - 'CORUM' for CORUM protein complexes
        - 'HP' for Human Phenotype Ontology

    :return:  Core output, csv file, the results of FEA and GSEA.
    """
    os.makedirs(save_dir, exist_ok=True)
    # OBO 文件使用同目录的 go-basic.obo，避免工作目录变化导致找不到文件
    obofilepath = str(Path(__file__).with_name("go-basic.obo").resolve())

    gene_list_path = []
    for dirpath, dirnames, filenames in os.walk(deg_res):
        if "significant_genes.xlsx" in filenames:
            file_path = os.path.join(dirpath, "significant_genes.xlsx")
            gene_list_path.append(file_path)

    if len(gene_list_path) == 0:
        raise ValueError("No valid gene list")

    for gene_list in gene_list_path:
        gene_list_p = Path(gene_list).resolve()
        # 约定：.../<feature_id>/result/significant_genes.xlsx
        feature_id = gene_list_p.parent.parent.name
        save_dir_feature = Path(save_dir) / feature_id
        save_dir_feature.mkdir(parents=True, exist_ok=True)
        geneDf = pd.read_excel(gene_list_p)

        generate_clinical_phenotype(str(gene_list_p), feature_id, clinical_information, clinical_col, str(save_dir),
                                    category)

        geneList = geneDf["Gene ID"].tolist()
        eGODataFrame = performFEA(geneList, sources=sources)
        # print(eGODataFrame)
        # Save results to a file
        eGODataFrame.to_csv(save_dir_feature / 'GO_enrichment_results(PyPathomics_fpkm).csv', index=False)
        # Generate DAG visualization
        goTerms = eGODataFrame["native"].head(1)
        highlightColor = "#C0392BFF"  # Highlight color
        performDAG(obofilepath, goTerms, highlightColor, str(save_dir_feature / "DAG_multiple_terms.png"))

        # Create bar plot visualization
        # df = pd.read_csv(os.path.join(save_dir, 'GO_enrichment_results(PyPathomics_fpkm).csv'))
        performCreateBarplot(
            df=eGODataFrame,
            xData='intersection_size',
            yData='native',
            numTop=5,
            colorBy='source',
            colors=["#8DA1CB", "#FD8D62", "#66C3A5"],
            title=f"{sources} functional enrichment",
            facetGrid='source',
            xAxisTitle="Intersection size",
            yAxisTitle="Terms",
            width=1000,
            height=500,
            save_dir=str(save_dir_feature)
        )

        # Generate term-gene network graph
        # GODf = pd.read_csv(os.path.join(save_dir, 'GO_enrichment_results(PyPathomics_fpkm).csv'))
        GODf = eGODataFrame.iloc[0:5]
        termsGeneNet(
            GODf,
            labelTypes=['terms', 'shared_genes'],
            title='Gene-term network graph',
            save_dir=str(save_dir_feature)
        )

        # Create UpSet plot
        # df = pd.read_csv(os.path.join(save_dir, 'GO_enrichment_results(PyPathomics_fpkm).csv'))
        df = eGODataFrame.iloc[0:5]
        plotUpSet(df, width=10, save_dir=str(save_dir_feature))

        # Generate ridge plot
        # enrichmentDf = pd.read_csv(os.path.join(save_dir, 'GO_enrichment_results(PyPathomics_fpkm).csv'))
        enrichmentDf = eGODataFrame.iloc[0:5]
        # geneListDf = pd.read_excel(geneListFilePath)
        wideFormatDf = createWideFormatDf(enrichmentDf, geneDf, fillNA=0)
        # print(wideFormatDf.head())
        generateRidgePlot(wideFormatDf, height=500, width=800, save_dir=str(save_dir_feature))

        # Perform GSEA analysis
        # geneDf = pd.read_excel(geneListFilePath)
        geneDf = geneDf[['Gene ID', 'LogFC']]
        sortedGeneDf = geneDf.sort_values(by='LogFC', ascending=False)
        preRES = performGSEA(sortedGeneDf, outdir=str(save_dir_feature / 'GSEA(PyPathomics_fpkm)'))

        # Generate GSEA plot
        index = 0  # Index of the term to plot
        term = preRES.res2d.Term[index]
        data = preRES.results[term]
        gseaPlot = GSEAPlotly(
            preRES=preRES,
            tag=data['hits'],  # Indices of hits
            runes=data['RES'],  # Running enrichment scores
            nes=data['nes'],  # Normalized enrichment score
            pval=data['pval'],
            fdr=data['fdr'],
            rankMetric=preRES.ranking,  # Gene ranking
            title=term,
            fileName=str(save_dir_feature / "GSEAPlotly.png")
        )
        gseaPlot.show()
        gseaPlot.savefig()

        # Generate heatmap
        # enrichmentDf = pd.read_csv(os.path.join(save_dir, 'GO_enrichment_results(PyPathomics_fpkm).csv'))
        enrichmentDf = eGODataFrame.iloc[0:5]
        # geneListDf = pd.read_excel(geneListFilePath)
        geneListDf = geneDf[['Gene ID', 'LogFC']]
        # geneListDf = geneListDf.iloc[0:200]
        wideFormatDf = createWideFormatDf(enrichmentDf, geneListDf)
        generateHeatmap(
            wideFormatDf=wideFormatDf,
            title='Gene Expression Heatmap by GO Terms',
            save_dir=str(save_dir_feature)
        )

    return


def main():
    parser = argparse.ArgumentParser(description="Functional enrichment analysis and gene set enrichment analysis")
    parser.add_argument("--deg_res", default="../../example_folder/Degs", type=Path, required=True, help="Path to the degs")
    parser.add_argument("--source", default="GO:All", type=str, required=True, help="Path to the bulk RNA")
    parser.add_argument("--clinical_information", default="../../example_folder/survival_info.csv", type=Path, required=True, help="Path to the clinical info")
    parser.add_argument("--clinical_col", default="survival_time", type=str, required=True, help="The clinical column: TNM_stage,ES_grade...")
    parser.add_argument("--enrich_save_dir", default="../../example_folder/FEA_GSEA", type=Path, required=True, help="Path of save directory")
    parser.add_argument("--proxy_port", default=7897, type=int, required=True)
    args = parser.parse_args()
    os.environ['HTTP_PROXY'] = f'http://127.0.0.1:{args.proxy_port}'
    os.environ['HTTPS_PROXY'] = f'http://127.0.0.1:{args.proxy_port}'
    print("Please enable the global system proxy: for example: os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897")
    enrichment_analysis(args.deg_res, args.enrich_save_dir, args.source, args.clinical_information, args.clinical_col)


if __name__ == "__main__":
    main()