'''
Author: Mai Oudah
Henschel Lab
'''

from __future__ import division
import os
import sys
import csv
import weka.core.jvm as jvm
import weka.core.serialization as serialization
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

# This script takes the output of the HFE method as an input and produces classification models and the evaluation result of each model, in terms of precision, recall, f-meature, AUC, etc., via WEKA framework.

#To call:
#python /home/HFE/scripts/weka_script.py /home/Desktop/dataset/hfe_result/Final_feature_set.csv

#The script argument is the complete directory of the final feature set.
 
def call_weka(file_dir, ml_opt, ofile_dir):

	loader = Loader(classname="weka.core.converters.CSVLoader")
	data = loader.load_file(file_dir)
	data.class_is_last()
	filtered = data

	ml_id = ''
	if ml_opt != '0':	
		if ml_opt == '1':
			classifier = Classifier(classname="weka.classifiers.functions.LibSVM", options=["-S", "0", "-K", "2", "-D", "3", "-G", "0.0", "-R", "0.0", "-N", "0.5", "-M", "40.0", "-C", "1.0", "-E", "0.001", "-P", "0.1", "-seed", "1"])
			ml_id = 'SVM'
		elif ml_opt == '3':
			classifier = Classifier(classname="weka.classifiers.functions.MLPClassifier", options=['-N', '2', '-R', '0.01', '-O', '1.0E-6', '-P', '1', '-E', '1', '-S', '1'])
			ml_id = 'MLPC'
		elif ml_opt == '4':
			classifier = Classifier(classname="weka.classifiers.trees.RandomForest", options=["-I", "100", "-K", "0", "-S", "1", "-num-slots", "1"])
			ml_id = 'RF'
		elif ml_opt == '2':
			classifier = Classifier(classname="weka.classifiers.meta.Bagging", options=["-P", "100", "-S", "1", "-I", "10", "-W", "weka.classifiers.trees.M5P",  "--", "-M", "4.0"])
			ml_id = 'BagM5P'
		elif ml_opt == '5':
			classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
			ml_id = 'J48'
		elif ml_opt == '7':
			classifier = Classifier(classname="weka.classifiers.functions.RBFNetwork", options=["-B", "2", "-S", "1", "-R", "1.0E-8", "-M", "-1", "-W", "0.1"])
			ml_id = 'RBFNet'
		elif ml_opt == '8':
			classifier = Classifier(classname="weka.classifiers.bayes.BayesNet", options=["-D", "-Q", "weka.classifiers.bayes.net.search.local.K2", "--", "-P", "1", "-S", "BAYES", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A", "0.5"])
			ml_id = 'BayesNet'
		elif ml_opt == '6':
			classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
			ml_id = 'NaiveBayes'
		elif ml_opt == '9':
			classifier = Classifier(classname="weka.classifiers.functions.SimpleLogistic", options=["-I", "0", "-M", "500", "-H", "50", "-W", "0.0"])
			ml_id = 'LogReg'
		filtered.class_is_last()
		evaluation = Evaluation(filtered)
		evaluation.crossvalidate_model(classifier, filtered, 10, Random(42))
		print "Evaluation: Done."

		ofile = open(ofile_dir+ml_id+"_results.txt", 'wb')

		print >> ofile, evaluation.summary()
		print >> ofile, evaluation.class_details().encode('ascii', 'ignore')
		print >> ofile, evaluation.matrix().encode('ascii', 'ignore')
		serialization.write(ofile_dir+ml_id+".model", classifier)
		print "Saving "+ml_id+" Model: Done."

		ofile.close()

if __name__ == '__main__':

	file_dir = sys.argv[1]

	file_name = file_dir.strip().split('/')[-1]
	f_dir = file_dir.strip().replace(file_name, '')
	ofile_dir = f_dir+file_name.split('.')[0]+"_weka_results/"	
	os.system("mkdir "+ofile_dir)
	jvm.start(max_heap_size="1024m", packages=True)

	for k in range(3, 10): #range(4, 7) for RF, DT and NB
		call_weka(file_dir, str(k), file_dir.strip().replace(file_name, file_name.split('.')[0])+"_weka_results/")
	jvm.stop()
