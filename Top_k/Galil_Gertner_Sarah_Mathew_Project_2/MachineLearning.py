#---- Imports -----------------------------------------------------------------

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

#---- Classes -----------------------------------------------------------------

class MachineLearning():

    def __init__(self, ad_reduced, nci_reduced, sc):
        self.ad_reduced = ad_reduced
        self.nci_reduced = nci_reduced
        self.sc = sc
        self.labeled_genes = None
        self.labeled_genes_and_age = None

        self.mean_gini_error = None
        self.mean_entropy_error = None
        self.mean_rf_error = None
        self.mean_rf_error_whole_data = None

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
        ad_genes_and_age = ad_no_cluster_ids \
                                   .map(lambda x:
                                        ([x[0][1][0]], x[1])) \
                                   .map(lambda x: x[0] + x[1])

        nci_genes_and_age = nci_no_cluster_ids \
                                    .map(lambda x:
                                        ([x[0][1][0]], x[1])) \
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

        ad_genes_and_age_labeled = ad_genes_and_age \
                                        .map(lambda features:
                                                    LabeledPoint(1, features))
                                   # AD gets a label of 1

        nci_genes_and_age_labeled = nci_genes_and_age \
                                        .map(lambda features:
                                                    LabeledPoint(0, features))
                                   # NCI gets a label of 0

        # Stack NCI and AD labels together

        all_groups_gene_vals_only = ad_gene_vals_only_labeled \
                                   .union(nci_gene_vals_only_labeled)

        all_groups_genes_and_age = ad_genes_and_age_labeled \
                                        .union(nci_genes_and_age_labeled)

        #import pdb; pdb.set_trace()

        self.labeled_genes = all_groups_gene_vals_only
        self.labeled_genes_and_age = all_groups_genes_and_age

    # Data name is a string indicating which model is being built so that
    # the model can be saved with the appropriate name.
    def build_model(self, model_type, number_of_trees):

        if (model_type == 'genes'):
            group = self.labeled_genes
        if (model_type == 'life_stats'):
            group = self.labeled_genes_and_age

        ntrees = number_of_trees

        #group = self.labeled_genes

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

        impurities  = ['gini', 'entropy']

        for impurity in impurities:

            set1_testErr = self.decision_tree(train_set_1,
                                                       test_set_1, impurity)
            set2_testErr = self.decision_tree(train_set_2,
                                                       test_set_2, impurity)
            set3_testErr = self.decision_tree(train_set_3,
                                                       test_set_3, impurity)

            print("Error on set 1 for {} impurity: {}".format(impurity,
                                                              set1_testErr))
            print("Error on set 2 for {} impurity: {}".format(impurity,
                                                              set2_testErr))
            print("Error on set 3 for {} impurity: {}".format(impurity,
                                                              set3_testErr))
            if impurity == 'gini':
                self.mean_gini_error = ((float(set1_testErr)
                                       + float(set2_testErr)
                                       + float(set3_testErr)) / 3)

            if impurity == 'entropy':
                self.mean_entropy_error = ((float(set1_testErr)
                                          + float(set2_testErr)
                                          + float(set3_testErr)) / 3)

            print("Mean {} error: {}".format(impurity,
                                                 ((float(set1_testErr)
                                                + float(set2_testErr)
                                                + float(set3_testErr)) / 3)))
            #import pdb; pdb.set_trace()

        set_1_rf_error = self.random_forest(train_set_1,
                                            test_set_1,
                                            'entropy',
                                            ntrees)

        set_2_rf_error = self.random_forest(train_set_2,
                                            test_set_2,
                                            'entropy',
                                            ntrees)

        set_3_rf_error =  self.random_forest(train_set_3,
                                             test_set_3,
                                             'entropy',
                                             ntrees)

        print("Error on set 1 for random forest: ", set_1_rf_error)
        print("Error on set 2 for random forest: ", set_2_rf_error)
        print("Error on set 3 for random forest: ", set_3_rf_error)

        self.mean_rf_error = (                  ((float(set_1_rf_error)
                                                + float(set_2_rf_error)
                                                + float(set_3_rf_error)) / 3 ))

        #import pdb; pdb.set_trace()


    def random_forest(self, train_sample, test_sample, impurity, num_trees):

        #import pdb; pdb.set_trace()
        rf_model = RandomForest.trainClassifier(train_sample,
                                          numClasses=2,
                                          categoricalFeaturesInfo={},
                                          numTrees=int(num_trees),
                                          featureSubsetStrategy="auto",
                                          impurity=impurity,
                                          maxDepth=5,
                                          seed=123)

        # Cross-validate on the model

        return self.cross_validate(rf_model, test_sample)


        # This is the cross validation where we predict the results of the
        # test sample based on the rf built on the training sample.


        #predictions = rf.predict(test_sample.map(lambda x: x.features))
        #labels_and_predictions = test_sample.map(lambda lp:



    def decision_tree(self, train_sample, test_sample, impurity):

        dt_model = DecisionTree.trainClassifier(train_sample,
                                                   numClasses=2,
                                                   categoricalFeaturesInfo={},
                                                   #impurity='gini',
                                                   impurity=impurity,
                                                   maxDepth=5)

        return self.cross_validate(dt_model, test_sample)


    def cross_validate(self, model, test_sample):
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


    def print_all_data(self):
        print("Mean gini impurity error: ", self.mean_gini_error)
        print("Mean entropy impurity error: ", self.mean_entropy_error)
        print("Mean random forest test sample error: ",
                                    self.mean_rf_error)
        #print("Mean random forest whole data error: ",
                                    #self.mean_rf_error_whole_data)










