from __future__ import division
import os
import sys
import csv
import math
from scipy.stats.stats import pearsonr
import numpy as np
import weka.core.jvm as jvm
import weka.core.serialization as serialization
from weka.filters import Filter
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection

#To call, e.g.:
#python /home/hfe_algorithm.py /home/otu_table.tab /home/label_vector.tab /home/tax_table.tab

#The 1st argument is the directory of the labeled tabulated otu table; 
#the 2nd argument is the directory of the labels file (tab-separated table), and
#the 3rd argument is the directory of the taxonomy file (tab-separated table). 

def into_tab(array, tab_dir, delim):
	ofilex = open(tab_dir, "wb")
	for i in range(len(array)):
		tmp = []
		for j in range(len(array[i])):
			tmp.append(array[i][j])
		ofilex.write(str(tmp).replace("['",'').replace("']", '').replace(", ", delim).replace("'", '')+'\n')
	ofilex.close()

def get_IG(ofile_dir):
	data = loader.load_file(ofile_dir)
	data.class_is_last()

	evaluator = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
	search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
	attsel = AttributeSelection()
	attsel.search(search)
	attsel.evaluator(evaluator)
	attsel.select_attributes(data)

	results = {}

	if attsel.number_attributes_selected < 2:
		flag = 0
		output = attsel.results_string
		for i in output.split('\n'):
			if (flag != 0):
				if len(i.split(' '))>2:
					t=[]
					for f in i.split(' '):
						if f!='':
							t.append(f)
					r_tax = ''
					for c in range(len(t)):
						if c>1:
							r_tax = r_tax+t[c]+' '
					results.update({str(r_tax.strip()): float(t[0].strip())})
				else:
					break
			if "Ranked attributes" in i:
				flag = 1
		mean_score = sum(results.values())/len(results.values())
		os.system("rm -r "+ofile_dir)
	else:
		results = dict([(str(data.attribute(attr[0]).name), attr[1]) for attr in attsel.ranked_attributes])
		mean_score = attsel.ranked_attributes[:,1].mean()
	
	return results, mean_score


#split a list to sublists
def split_list(dataset, n):
	return [ dataset[i:i+n] for i in xrange(0, len(dataset), n) ]
	

## Make sure that there is no redundancy in the tax file

if __name__ == '__main__':

		
	initial_FS_dir = sys.argv[1] 
	labels_dir = sys.argv[2]
	tax_dir = sys.argv[3]

	file_name = initial_FS_dir.strip().split('/')[-1]
	f_dir = initial_FS_dir.strip().replace(file_name, '')
	results_dir = f_dir+"hfe_"+file_name.split('.')[0]+'/'
	os.system("mkdir "+results_dir)
	os.system("chmod -R 777 "+results_dir)
	
	corr_method = 1   #1: pearson correlation 

	root_node = ''
	tax_type = '0'
	if tax_type == '0':
		root_node = "k__Bacteria"

	# dictionary of the dataset (each column represents a sample, while each row is for a feature (e.g. an OTU))
	dataset = np.loadtxt(initial_FS_dir, delimiter='\t', dtype= object)
	feature_records = { str(line[0]):[float(i) for i in line[1:]] for line in dataset }
	
	labels = np.loadtxt(labels_dir, delimiter='\t', dtype=object)
	label_vector = { labels[0]: list(labels[1:]) }

	feature_records.update(label_vector)

	print "Done with loading from the OTU table!"

	# dictionary of the taxonomy file
	tax_data = np.loadtxt(tax_dir, delimiter='\t', dtype=object)
	tax_file = { line[0]:list(line[1:]) for line in tax_data }

	# assuming that the columns are for each level in the hierarchy in addition to the leafs' labels.
	no_of_levels = len(tax_file[tax_file.keys()[0]])
	print "Number of levels: ", no_of_levels

	# What will indicate an empty level
	empty = []
	if tax_type == '0':
		empty = [3,3,3,3,3,3,3]

	child_parent_dict = {}
	leaf_name_set = set()
	taxon_value_dict = {}
	otu_parent_dict = {}
	#parsing the tax file and prepare the data
	for element in tax_file:
		tax = tax_file[element]	# assume that the line starts with the leaf label (e.g. otu_id) then the path details.
		# set the name of every leaf (i.e. labeling each leaf in the hierarchy with otu_id for example)
		if root_node in tax:
			leaf_indx = -1
			if tax_type == '0':
				for i in range(1, no_of_levels):
					if len(tax[i].strip()) == empty[i]:
						leaf_indx = i-1
						break	
			if leaf_indx == -1:
				leaf_indx = no_of_levels-1
			if not(element in leaf_name_set):
				otu_parent_dict.update({element: tax[leaf_indx].strip()})
				taxon_value_dict.update({element: feature_records[element]})
				if tax_type == '0':
					if 's__' in tax[leaf_indx].strip():
						leaf_name_set.add(element)
						child_parent_dict.update({element: tax[leaf_indx].strip()})
					else:
						leaf_name_set.add(tax[leaf_indx].strip())			

			for i in range(1, no_of_levels):
				#print "index: "+str(i)
				#for each level, specify the parent of each child (assuming each child has one parent)
				if tax_type == '0':
					if not(tax[i].strip() in child_parent_dict.keys()) and not(len(tax[i].strip()) == empty) and not(len(tax[i-1].strip()) == empty):
						child_parent_dict.update({tax[i].strip(): tax[i-1].strip()})

			# assign each taxon an abundance vector
			for i in range(no_of_levels):
				if not(tax[i].strip() in taxon_value_dict.keys()) and not(len(tax[i].strip()) == empty):
					taxon_value_dict.update({tax[i].strip(): feature_records[element]})
				elif tax[i].strip() in taxon_value_dict.keys():
					merged_values = list(np.sum([feature_records[element],taxon_value_dict[tax[i].strip()]], axis=0))
					taxon_value_dict[tax[i].strip()] = merged_values		


	print "Done with loading Hierarchy from the Tax table!"

	###########################################################
	   ## The First phase of Hierarchy selection approach ##
	###########################################################

	ofile1 = open(results_dir+'correlations.tab', "wb")
	ofile1.write('Pair of taxons'+'\t'+"correlation coefficient"+'\n')
	ofile2 = open(results_dir+'removed_taxa.tab', "wb")
	ofile2.write('leaf'+'\t'+"taxon"+'\n')
	
	removed_leaf_list = []
	leaves_to_add = []
	nodes_to_remove = []
	subtract_list = []
	
	leaf_name_list = list(leaf_name_set)
	for leaf in leaf_name_list:
		leaf_values = []
		if not (root_node in leaf):

			leaf_values = taxon_value_dict[leaf] 
		
			# The leaf's taxon and parent (recall: assuming there is only one parent)
			child_tax = leaf
			parent_tax = ''
			if not(root_node in child_parent_dict[child_tax]):
				parent_tax = child_parent_dict[child_tax]
				
				#correlation threshold
				corr_threshold = 0.7
				
				similarity = 0.0
			
				# calculate the correlation between the child and parent
				if corr_method == 1:
					# pearsonr' output (correlation, p-value)
					similarity = pearsonr(taxon_value_dict[child_tax], taxon_value_dict[parent_tax])[0]
					ofile1.write(parent_tax+'_'+child_tax+'\t'+str(similarity)+'\n')	
	

				# check the similarity value
				if (abs(similarity) > corr_threshold) or (similarity=="nan"):
					#remove the leaf
					removed_leaf_list.append(child_tax)
					nodes_to_remove.append(child_tax)
					leaf_name_list.remove(child_tax)
					leaf_name_set.remove(child_tax)
		
					ofile2.write(leaf+'\t'+child_tax+'\n')		
					if not(parent_tax in leaf_name_set):
							leaf_name_set.add(parent_tax)
							leaf_name_list.append(parent_tax)

							leaves_to_add.append(parent_tax)
				else:
					#to subtract the abundance of the child from the parent's abundance 
					subtract_list.append((parent_tax, child_tax))
		else:
			leaf_name_list.remove(leaf)
			leaf_name_set.remove(leaf)
			nodes_to_remove.append(leaf)

	print "Number of left leaves from phase 1: "+str(len(leaf_name_set))
	ofile3 = open(results_dir+'leaves_phase1.tab', "wb")
	ofile3.write('leaf_name'+'\n')
	for l in leaf_name_set:
		ofile3.write(l+'\n')
	ofile3.close()
	ofile1.close()
	ofile2.close()

	# create the temp feature set
	feature_value_FS = []
	for node in leaf_name_set:
		feature_value_FS.append([node]+taxon_value_dict[node])
	feature_value_FS.append(["label"]+feature_records["label"])
	transposed_table = zip(*feature_value_FS)
	ofile_dir = results_dir+'temp_feature_set1.csv'
	into_tab(np.array(transposed_table), ofile_dir, ',')

	###########################################################
	   ## The Second phase of Hierarchy selection approach ##
	###########################################################	

	jvm.start(max_heap_size="1024m", packages=True)
	loader = Loader(classname="weka.core.converters.CSVLoader")	
	
	path_dict = {}
	nodes_to_keep = []
	IG_dict = {}
	#not to print a feature more than once
	IG_phase1_features = []
	phases_features = {}
	for leaf in leaf_name_set:
		leaf_path = []
		leaf_taxon = leaf		

		if leaf_taxon in child_parent_dict.keys():
			leaf_path.append(leaf_taxon)

			parent_tax = child_parent_dict[leaf_taxon]
			leaf_path.append(parent_tax)
			
			for i in range(no_of_levels):
				if leaf_path[-1] in child_parent_dict.keys():
					if not(child_parent_dict[leaf_path[-1]] in leaf_path):
						parent_tax = child_parent_dict[leaf_path[-1]]
						leaf_path.append(parent_tax)
				else:
					break

			#the active nodes in the path from the leaf to the root
			selected_node_LP = []
			feature_value_LP = []

			for n in leaf_path:
				if (n in leaf_name_set) and (not(root_node in n)):
					selected_node_LP.append(n)
					feature_value_LP.append([n]+taxon_value_dict[n])

			if len(selected_node_LP)>0:
				#to compute the information gain via weka
				feature_value_LP.append(["label"]+feature_records["label"])
				transposed_table = zip(*feature_value_LP)
				ofile_dir = results_dir+'temp_table.csv'
				into_tab(np.array(transposed_table), ofile_dir, ',')
	
				results, mean_IG = get_IG(ofile_dir)
				#check the IG score of each selected node in the path
				for s in selected_node_LP:
					if not(s in IG_phase1_features):
						IG_phase1_features.append(s)
						phases_features.update({s: str(results[s])})

					if (results[s] < mean_IG) or (results[s] <= 0):
						nodes_to_remove.append(s)
					else:
						t = results[s]
						nodes_to_keep.append(s)
						if not(s in IG_dict.keys()):
							#save the survived nodes with their IG scores
							IG_dict.update({s: results[s]})
							phases_features.update({s: str(t)})

	print "Done with phase 2!"

	# remove what needs to be removed
	for r in nodes_to_remove:
		if not(r in nodes_to_keep) and (r in leaf_name_set):
			leaf_name_set.remove(r)

	print "Number of left features from phase 2: "+str(len(leaf_name_set))

	# create the temp feature set
	feature_value_FS = []
	for node in leaf_name_set:
		if not(root_node in node):
			feature_value_FS.append([node]+taxon_value_dict[node])
	feature_value_FS.append(["label"]+feature_records["label"])
	transposed_table = zip(*feature_value_FS)
	ofile_dir = results_dir+'temp_feature_set2.csv'
	into_tab(np.array(transposed_table), ofile_dir, ',')

	###########################################################
	   ## The third phase of Hierarchy selection approach ##
	###########################################################

	#calculate the avg. IG score of the nodes selected by the second phase
	IG_threshold = sum(IG_dict.values())/len(IG_dict.values())
	otus_to_keep = {}
	feature_value_LP = []
	valid_otu_list = []
	IG_results = {}
	for otu in otu_parent_dict.keys():
		if not(otu in nodes_to_remove) and not(otu in leaf_name_set):
			feature_value_LP.append([otu]+taxon_value_dict[otu])
			valid_otu_list.append(otu)			
	#to optimize the time needed
	otu_table_partitions = split_list(feature_value_LP, 4)
	for partition in otu_table_partitions:
		if len(partition)!=0:
			partition.append(["label"]+feature_records["label"])
			#mydata = np.array(partition)
			#transposed_table = mydata.T
			transposed_table = zip(*partition)
			ofile_dir = results_dir+'temp_table.csv'
			into_tab(np.array(transposed_table), ofile_dir, ',')
			results, avg_IG = get_IG(ofile_dir)
			IG_results.update(results)
	phase3_features = []		
	#check the IG score of the selected otu against the overall avg. IG
	for s in IG_results:
		if (IG_results[s] > (1.0*IG_threshold)) and (IG_results[s] > 0):
			otus_to_keep.update({s: IG_results[s]})
			phase3_features.append(s)
			phases_features.update({s: str(IG_results[s])})
	print "Done with phase 3!"
	print "Number of considered OTUs via phase 3: "+str(len(phase3_features))

	# create the temp feature set
	feature_value_FS = []
	for node in phase3_features:
		feature_value_FS.append([node]+taxon_value_dict[node])
	feature_value_FS.append(["label"]+feature_records["label"])
	transposed_table = zip(*feature_value_FS)
	ofile_dir = results_dir+'temp_feature_set3.csv'
	into_tab(np.array(transposed_table), ofile_dir, ',')

	######################################################################

	jvm.stop()
	
	# add informative OTUs
	for otu in otus_to_keep.keys():
		leaf_name_set.add(otu)

	#print the final feature list with IG scores
	ofile_IG = open(results_dir+'Final_feature_list_with_IG_scores.tab', "wb")
	ofile_IG.write('Feature'+'\t'+'IG_score'+'\n')
	for l in leaf_name_set:
		if not(root_node in l):
			ofile_IG.write(l+'\t'+phases_features[l]+'\n')
	ofile_IG.close()
	
	# print out the final list of leaves in a file
	ofile = open(results_dir+'Final_feature_list.tab', "wb")
	ofile.write('leaf_name'+'\n')
	for l in leaf_name_set:
		if not(root_node in l):
			ofile.write(l+'\n')
	ofile.close()
	# create the final feature set
	feature_value_FS = []
	for node in leaf_name_set:
		if not(root_node in node):
			feature_value_FS.append([node]+taxon_value_dict[node])
	print "Number of features in the final feature set: "+str(len(feature_value_FS))
	feature_value_FS.append(["label"]+feature_records["label"])
	transposed_table = zip(*feature_value_FS)
	ofile_dir = results_dir+'Final_feature_set.csv'
	into_tab(np.array(transposed_table), ofile_dir, ',')

	os.system("chmod -R 777 "+results_dir)

