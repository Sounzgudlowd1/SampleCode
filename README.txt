This is the main analysis done for 
http://35.243.182.41/final_project

It contains the main source files:
DFS--The class for constructing a DFS model with some basic reporting and metric creation.
ElasticNetRegularizer--The class for constructing a weighted sum of l1 norm and l2 norm
OneToOne--The actual input layer that transforms the model from a standard multi-layer perceptron to a DFS model
rna_dfs--the code that calls the above code and runs the analysis on the rna genomic sequence data for cancer prediction

To run I highly recommend using Keras with a Tensorflow backend using Python 3.6.  At the time of this writing 3.7 did not work 
with Keras.  I also recommend having Anaconda installed.