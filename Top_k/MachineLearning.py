#---- Imports -----------------------------------------------------------------

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

#---- Classes -----------------------------------------------------------------

class MachineLearning():

    def __init__(self, ad_reduced, nci_reduced, sc):
        self.ad_reduced = ad_reduced
        self.nci_reduced = nci_reduced
        self.sc = sc
        self.labeled_genes = None
        self.labeled_genes_and_life_stats = None


    def set_labeled_points_data(self):
        #import pdb; pdb.set_trace()
        # take out the cluster ids and only leave the values
        # note, cluster ids of top 10 t-test values are no longer
        # in rank order, but they are consistently ordered between
        # the patients so doesn't matter for the purposes of the model

        ad_no_cluster_ids = self.ad_reduced \
                                .map(lambda x:
                                    (x[0],
                                    [x[1][i] for i in range (1, 20, 2)]))
                                    # range gets only odd numbered values
                                    # from 1 - 20 in values section of
                                    # (key, value) pair.

        nci_no_cluster_ids = self.nci_reduced \
                                .map(lambda x:
                                    (x[0],
                                    [x[1][i] for i in range (1, 20, 2)]))
                                    # range gets only odd numbered values
                                    # from 1 - 20 in values section of
                                    # (key, value) pair.

        # Get rdds with only the gene vals of each control group
        ad_gene_vals_only = ad_no_cluster_ids.map(lambda x: x[1])
        nci_gene_vals_only = nci_no_cluster_ids.map(lambda x: x[1])
        #import pdb; pdb.set_trace()
        # Get rdds with both patient stats and gene vals for each control group
        ad_gene_and_life_stats = ad_no_cluster_ids \
                                   .map(lambda x:
                                        ([x[0][1][0],
                                          x[0][1][1],
                                          x[0][1][2]], x[1])) \
                                   .map(lambda x: x[0] + x[1])

        nci_gene_and_life_stats = nci_no_cluster_ids \
                                    .map(lambda x:
                                        ([x[0][1][0],
                                          x[0][1][1],
                                          x[0][1][2]], x[1])) \
                                    .map(lambda x: x[0] + x[1])

        # Transform rdds to Labeled Points for MLlib
        ad_gene_vals_only_labeled = ad_gene_vals_only \
                                   .map(lambda features:
                                               LabeledPoint(1, features))
                                   # AD gets a label of 1

        nci_gene_vals_only_labeled = nci_gene_vals_only \
                                    .map(lambda features:
                                               LabeledPoint(0, features))
                                   # NCI gets a label of 0

        ad_gene_and_life_stats_labeled = ad_gene_and_life_stats \
                                        .map(lambda features:
                                                    LabeledPoint(1, features))
                                   # AD gets a label of 1

        nci_gene_and_life_stats_labeled = nci_gene_and_life_stats \
                                        .map(lambda features:
                                                    LabeledPoint(0, features))
                                   # NCI gets a label of 0

        # Stack NCI and AD labels together

        all_groups_gene_vals_only = ad_gene_vals_only_labeled \
                                   .union(nci_gene_vals_only_labeled)

        all_groups_gene_and_life_stats = ad_gene_and_life_stats_labeled \
                                        .union(nci_gene_and_life_stats_labeled)


        self.labeled_genes = all_groups_gene_vals_only
        self.labeled_genes_and_life_stats = all_groups_gene_and_life_stats

    # Data name is a string indicating which model is being built so that
    # the model can be saved with the appropriate name.
    def build_model(self):

        #if (model_type == 'genes'):
        #    group = self.labeled_genes
        #if (model_type == 'life_stats'):
        #    group = self.labeled_genes_and_life_stats

        group = self.labeled_genes

        # Create a decision tree for both Labeled point groups
        # With the life stats and without
        # IMPORTANT NOTE: Python 3.5 and Python 2.7 work differently on lambda
        # functions using multiple variables.
        # Where python 2.7 uses (lambda v, p: v != p)
        #       pythin 3.5 uses (lambda v_p: v_p[0] != v_p [1])
        # Creating a seed value, radomly chosen as 1234 in order to be able to
        # recreate the exact same values every time.

        (train_genes, sample3_genes) = group.randomSplit([0.67, 0.33], 1234)
        (sample1_genes, sample2_genes) = train_genes.randomSplit([0.5, 0.5], 1234)

        train_set_1 = sample1_genes.union(sample2_genes)
        test_set_1 = sample3_genes

        train_set_2 = sample2_genes.union(sample3_genes)
        test_set_2 = sample1_genes

        train_set_3 = sample1_genes.union(sample3_genes)
        test_set_3 = sample2_genes


        set1_testErr = self.train_and_test_samples(train_set_1, test_set_1)
        set2_testErr = self.train_and_test_samples(train_set_2, test_set_2)
        set3_testErr = self.train_and_test_samples(train_set_3, test_set_3)

        print("Error on set 1: ", set1_testErr)
        print("Error on set 2: ", set2_testErr)
        print("Error on set 3: ", set3_testErr)

        import pdb; pdb.set_trace()

        #model = DecisionTree.trainClassifier(group, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5)
#predictions = model.predict(group.map(lambda x: x.features))
#l_p=group.map(lambda lp: lp.label).zip(predictions)
#overallErr=l_p.filter(lambda v_p: v_p[0] != v_p[1]).count()/float(group.count())
#print('Error = '+ str(overallErr)) You are going to see that the error is 0.22 which is much lower than what we see in cross validation. While it's expected for overall error to be lower relative to CV errors, in some samples it's 2 times less which indicated that our current model isn't that good.
# print(model.toDebugString())

### now I want to see how the model performs with impurity='entropy'
#model = DecisionTree.trainClassifier(group, numClasses=2, categoricalFeaturesInfo={}, impurity='entropy', maxDepth=5) there are fewer nodes in this model. Giniis known for overfitting.
#predictions = model.predict(group.map(lambda x: x.features))
#l_p=group.map(lambda lp: lp.label).zip(predictions)
#overallErr=l_p.filter(lambda v_p: v_p[0] != v_p[1]).count()/float(group.count())
#print('Error = '+ str(overallErr)) Error is higher 0.25
# You'll need to add part that does cross validation. Record the values of TestError for each sample and compare them:
# 1) with the overall Error
# 2) with the model that uses impurity gini
#from pyspark.mllib.tree import RandomForest, RandomForestModel
#rf = RandomForest.trainClassifier(group, numClasses=2, categoricalFeaturesInfo={}, numTrees=50, featureSubsetStrategy="auto", impurity='gini', maxDepth=5) I started with 50 trees,  the higher the number of trees, the lower the variance which ultimately leads to higher accuracy. featureSubsetStrategy asks us to set the number of features to use for splitting at each tree node. I have no preference, so I set it to default.
#predictions=rf.predict(group.map(lambda x: x.features))
#l_p=group.map(lambda lp: lp.label).zip(predictions)
#overallErr=l_p.filter(lambda v_p: v_p[0] != v_p[1]).count()/float(group.count())
#print(overallErr)
# let's try 100 trees
#rf = RandomForest.trainClassifier(group, numClasses=2, categoricalFeaturesInfo={}, numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=5)
#predictions=rf.predict(group.map(lambda x: x.features))
#l_p=group.map(lambda lp: lp.label).zip(predictions)
#overallErr=l_p.filter(lambda v_p: v_p[0] != v_p[1]).count()/float(group.count())
#I want to check some other metrics to evaluate the performance. I am especially interested in ROC. It's easy to understand if you read about it.
#from pyspark.mllib.evaluation import BinaryClassificationMetrics
#metrics = BinaryClassificationMetrics(l_p)
#print(metrics.areaUnderROC) so this AUC varies from 0.5 and 1, where 0.5 would be random guessing and 1 is a model that is 100% accurate. We want this value to be closer to 1. 0.81 is quite decent AUC value, but again it has to be tested on cross validation sample.




    def train_and_test_samples(self, train_sample, test_sample):

        model = DecisionTree.trainClassifier(train_sample,
                                                   numClasses=2,
                                                   categoricalFeaturesInfo={},
                                                   impurity='gini',
                                                   maxDepth=5)

        predictions = model.predict(test_sample.map(lambda x: x.features))
        labelsAndPredictions = test_sample.map(lambda lp: lp.label) \
                                         .zip(predictions)

        # Find all values where value and prediction don't match in test group
        # and divide by total number of vlaues in test group: yields error.
        testErr = labelsAndPredictions.filter(lambda v_p: v_p[0] != v_p[1]) \
                                      .count() / float(test_sample.count())

        print("Test error for gene model: ", testErr)

        print("Learned classification tree model: ")
        print(model.toDebugString())
        #model.save(self.sc, "model_{}" .format(model_type))

        # Get true-negative, true-positive, false-negative, false-positives
        # for gene_only model

        genes_TN = labelsAndPredictions.filter(lambda v_p: v_p[0] == 0.0
                                                       and v_p[1] == 0.0)

        genes_TP = labelsAndPredictions.filter(lambda v_p: v_p[0] == 1.0
                                                       and v_p[1] == 1.0)


        genes_FP = labelsAndPredictions.filter(lambda v_p: v_p[0] == 0.0
                                                       and v_p[1] == 1.0)

        genes_FN = labelsAndPredictions.filter(lambda v_p: v_p[0] == 1.0
                                                       and v_p[1] == 0.0)

        TN_percent = genes_TN.count() / float(test_sample.count())
        TP_percent = genes_TP.count() / float(test_sample.count())
        FP_percent = genes_FP.count() / float(test_sample.count())
        FN_percent = genes_FN.count() / float(test_sample.count())


        print("True negative: ", TN_percent)
        print("True positive: ", TP_percent)
        print("False positive: ", FP_percent)
        print("False negative: ", FN_percent)

        TN_count = genes_TN.count()
        TP_count = genes_TP.count()
        FP_count = genes_FP.count()
        FN_count = genes_FN.count()


        print("True negative: ", TN_count)
        print("True positive: ", TP_count)
        print("False positive: ", FP_count)
        print("False negative: ", FN_count)

        return testErr














