# Hierarchical Feature Engineering
## Synopsis
We developed an algorithm that deals with Hierarchical Feature Spaces, with focus on relative abundances of microbial communities but with the potential to be expanded to a wide range of Machine Learning tasks were features have an underlying hierarchical structure. In our case, the "initial" feature space are Operational taxonomic units (OTUs). For all OTUs, we identify their taxonomic lineage using standard tools such as RDP. We subsequently calculate relative abundance vectors for all internal nodes of the hierarchy (here: NCBI taxonomy). These are the new features that also will be considered during the learning process, since they represent abstraction (generalization) that make sense from an evolutionary perspective. Phylogenetic clades commonly share genetically encoded traits that are likely to be informative. 
After feature generation, we subject all (new and generated) features to filtering:
1. redundancy reduction
2. filtering by information gain
Both steps are part of our heuristic feature subset selection, using the taxonomy as a guidance for the heuristic: each step filters along the pathes from the taxonomy root to each leave (as compared to an exhaustive global comparison). This way, the algorithm operates in linear time O(n), where n is the number of OTUs.

Please use the data format (csv) as in the provided sample data. 

## Dependencies: 
Python 2.7, python-weka-wrapper, numpy

## Data
We also provide datasets, which we used in the publication (under review), see data directory.


