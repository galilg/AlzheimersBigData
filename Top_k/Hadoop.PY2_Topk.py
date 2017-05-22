#!/bin/python

#----Imports ------------------------------------------------------------------

from pyspark import SparkContext, SparkConf
import math
import pprint
import time

#---- Classes -----------------------------------------------------------------

class Top_k():

    def __init__(self):
        self.conf = SparkConf().setMaster("local[4]").setAppName("Get top-k")
        self.sc = SparkContext(conf=self.conf)

        self.rosmap_file = None
        self.gene_file = None

        # Top k and cluster values
        self.top_k = None
        self.allInfo = None
        self.allInfo_keyed_by_cluster = None
        self.cluster_vals = None

        # For machine learning
        self.nci_clusters = None
        self.ad_clusters = None

        # Machine learning patient top-k cluster scores with other stats
        self.nci_recuded = None
        self.ad_reduced = None

        self.ros_2_cluster = None


    def find_top_k(self, rosmap_file, gene_cluster_file):

        # Load HDFS files into a SparkContext
        self.load_files_from_hdfs(gene_cluster_file, rosmap_file)

        # Transition rosmap and gene_cluster files into (pos, cluster) pairs
        location_cluster = self.map_column_position_to_cluster()

        # Gather NCI statistics and number of NCI patients
        stat_nci, nci_count, nci_cluster_vals = self.specific_stat \
                                                        (self.rosmap,
                                                         location_cluster,
                                                         ['1'])

        # Gather AD statistics and number of AD patients
        stat_ad, ad_count, ad_cluster_vals = self.specific_stat(self.rosmap,
                                                    location_cluster, ['4','5'])

        # save cluster_vals for machine learning
        self.nci_clusters = nci_cluster_vals #.map(lambda x: (0, x))
        self.ad_clusters = ad_cluster_vals #.map(lambda x: (1, x))
        #self.cluster_vals = nci_clusters.union(ad_clusters)

        # join the statistical info for ad and nci
        # map: create (cluster, ((ad avg, ad std), (nci avg, nci std)))
        # example ('190', ((19.540666666666663, 5.394197046621285),
        #                 (19.540666666666663, 5.394197046621285)))
        # map(continue): do t-test on statistical information,
        # create (cluster, t-test)
        allInfo = stat_ad.join(stat_nci) \
                  .map(lambda x: (x[0], (x[1][0][0]-x[1][1][0])
                                       / math.sqrt((x[1][0][1]**2)
                                       / ad_count
                                       + (x[1][1][1]**2)/nci_count),
                                          x[1][0], x[1][1]))
        # x is the list of the top 10 values according to t-test values

        # Store allInfo to do lookups on
        self.allInfo = allInfo
        self.allInfo_keyed_by_cluster = allInfo.map(lambda x: (x[0], (x[1:])))

        # Store top k as class variable
        self.top_k = allInfo.top(10, lambda x: x[1])


    def load_files_from_hdfs(self, gene_cluster_file, rosmap_file):
        # import the gene_cluster file and separate values
        # filter for only humans
        self.gene_file = self.sc.textFile(gene_cluster_file, 8) \
                        .map(lambda genes: genes.split(',')) \
                        .filter(lambda organism: organism[2] == "Human")

        # import the patient file and separate values
        self.patient_info = self.sc.textFile('patients.csv', 8) \
                              .map(lambda x: x.split(','))

        # map: Split the rosmap file by commas:
        self.rosmap = self.sc.textFile(rosmap_file, 8) \
                      .map(lambda items: items.split(','))


    def map_column_position_to_cluster(self):
        # (cluster, [entrez, entrez,...,])
        cluster_and_entrez_ids = self.gene_file \
                                .map(lambda row: (row[0], row[4]))
                                # (cluster, [entrez, entrez,...,])

        # flatMap: make values with the key mapping to each value in the list
        cluster_2_entrez = cluster_and_entrez_ids \
                          .flatMapValues(lambda x: x.split(';'))
                          # (cluster, entrez)

        # map: switch key and value places
        entrez_2_cluster = cluster_2_entrez \
                          .map(lambda val: (val[1], val[0]))
                          # (entrez, cluster)


        # filter the header row from rosmap
        # map: create the key as 'PATIENT_ID',
        # create a list of pairs of entrez ids and their position
        # flatmap: create values where the key is the patient_id and
        # the value is each tuple in the list
        patient_entrez_position = self.rosmap \
                                    .filter(lambda x: x[0] == 'PATIENT_ID') \
                                    .map(lambda x: (x[0],
                                                    [(x[i],i)
                                                    for i in range(len(x))])) \
                                    .flatMapValues(lambda x:x)
                                    # ('PATIENT_ID", (entrez, pos))

        #import pdb; pdb.set_trace()
        # map: remove the patient_id as the key
        entrez_position = patient_entrez_position \
                            .map(lambda x: x[1]) # (entrez, pos)

        # join our rosmap data and cluster data based on entrez
        entrez_cluster_position = entrez_2_cluster.join(entrez_position)
                                # (entrez, (cluster, location))

        # map: remove entrez id, switch location and cluster
        location_cluster =  entrez_cluster_position \
                                .map(lambda entry: (entry[1][1],
                                                    entry[1][0]))
                                                    # (location, cluster)

        return location_cluster


    def print_cluster_values(self, cluster):

        lookup = self.allInfo_keyed_by_cluster.lookup(cluster)
        print("Cluster: ", cluster)
        print("Cluster T-test score: ", lookup[0][0])
        print("AD values for cluster (mean, std): ", lookup[0][1])
        print("NCI values for cluster (mean, std): ", lookup[0][2])


    def print_first_two_AD_patients_with_life_stats(self):
        pprint.pprint(self.ad_reduced.take(2))


    def print_top_k(self):
        for cluster in range(0, len(self.top_k)):
            print("Rank #%d" %(cluster + 1))
            print("Cluster: ", str(self.top_k[cluster][0]))
            print("T-test score: ", self.top_k[cluster][1])
            print("AD Values (Mean, STD): ", self.top_k[cluster][2])
            print("NCI Values (Mean, STD): %s \n" %(self.top_k[cluster][3],))

        #import pdb; pdb.set_trace()


    def specific_stat(self, rosmap, location_cluster, values):

        # filter out diagnosis of 1
        # map: key as patient id, value as list of entrez location and
        # associated value
        lines = self.rosmap.filter(lambda x: x[1] in values) \
                          .map(lambda x: (x[0],
                                          [(i, x[i])
                                          for i in range(2,len(x))]))
        count = lines.count()

        # flatMap: create values where the key is patient_id and values
        # are (location, gene_value)
        # map: remove the patient_id, key is now location and values are
        # (patient_id, gene_value)
        group = lines.flatMapValues(lambda x: x) \
                       .map(lambda x: (x[1][0], (x[0], x[1][1])))
                       # (location, (patient_id, gene_value))


        # join the rosmap nci data and cluster data,
        # create (location, (cluster, value))
        # map: ((cluster, patient_id), gene_value)
        # reduceby summing the gene_values for cluster for the same row,
        # example: (('2753', 'X332_120501'), 6.51)
        ros_2_cluster = location_cluster.join(group) \
                                            .map(lambda x: ((x[1][0],
                                            x[1][1][0]), x[1][1][1])) \
                                            .reduceByKey(lambda x, y:
                                                         float(x)+float(y))

        self.ros_2_cluster = ros_2_cluster
        # map: the key is the cluster id, value is the combined gene_value,
        # example: #('2753', 6.51)
        ros_2_cluster_paired = ros_2_cluster.map(lambda x: (x[0][0], x[1]))

        # map: the key is the cluster id, value is the combined gene_value in
        # a list, example: #('2753', [6.51])
        # reduceby: the key is the cluster id, value is the list of summed
        # cluster gene values
        ros_2_cluster_list = ros_2_cluster.map(lambda x: (x[0][0], [x[1]])) \
                                                  .reduceByKey(lambda x,y: x+y)

        # reduceby: sum the gene values, create (cluster, summed cluster gene
        # values)
        # map: find the average, create (cluster, average),
        # example: #('190', 0.0011929588929588926)
        # join the (cluster, average) with
        #          (cluster, [summed cluster gene values])
        # map: find the squared difference between summed cluster gene values
        # and the average
        # map: sum the differences and divide by the number of summed cluster
        # gene values
        # map: find the squareroot, create (cluster, (average, std))
        # example: ('4158', (1632.2369444444446, 392.986880982753))
        stat = ros_2_cluster_paired.reduceByKey(lambda x,y: float(x)+float(y)) \
                                           .map(lambda x: (x[0], x[1]/count)) \
                                           .join(ros_2_cluster_list) \
                                           .map(lambda x: (x[0], (x[1][0],
                                           [(float(i)-float(x[1][0]))**2
                                           for i in x[1][1]]))) \
                                           .map(lambda x: (x[0],
                                                          (x[1][0],
                                                           sum(x[1][1])
                                                         / len(x[1][1])))) \
                                           .map(lambda x: (x[0],
                                                          (x[1][0],
                                                          math.sqrt(x[1][1]))))


        return [stat, count, ros_2_cluster]


    def top_k_vals_for_patients(self):
        patient_background = self.patient_info.map(lambda x: (x[0], x[1:]))
        patient_stats = patient_background \
                       .map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2])))
                       # (ID, (age, gender edu))

        #import pdb; pdb.set_trace()
        pprint.pprint(patient_stats.take(10))
        #nci_vals = cluster_vals.filter(lambda x: x[0] == 0) \
        #                       .map(lambda y: (y[1][0], y[1][1]))

        #ad_vals = cluster_vals.filter(lambda x: x[0] == 1) \
        #                      .map(lambda y: (y[1][0], y[1][1]))

        nci_vals = self.nci_clusters
        ad_vals = self.ad_clusters
        nci_vals_cleaned = nci_vals.map(lambda x: (x[0][0], ([x[0][1], x[1]])))

        ad_vals_cleaned = ad_vals.map(lambda x: (x[0][0], ([x[0][1], x[1]])))

        top_k_vals = self.sc.parallelize(self.top_k)

        # Get the cluster values for patients for only the top 10
        # Remove t-test value

        nci_patient_cluster_vals_for_top_k = top_k_vals \
                                            .leftOuterJoin(nci_vals_cleaned) \
                                            .map(lambda x: (x[0], x[1][1]))
        ad_patient_cluster_vals_for_top_k = top_k_vals \
                                           .leftOuterJoin(ad_vals_cleaned) \
                                           .map(lambda x: (x[0], x[1][1]))

        nci_patients = nci_patient_cluster_vals_for_top_k.map(lambda x:
                                                             (x[1][0],
                                                              [x[0], x[1][1]]))

        ad_patients = ad_patient_cluster_vals_for_top_k.map(lambda x:
                                                           (x[1][0],
                                                           [x[0], x[1][1]]))


        #return nci_patients, ad_patients
        nci_patients_w_background = nci_patients.leftOuterJoin(patient_stats) \
                                                .map(lambda x: x)
            #.map(lambda x: ((x[0], x[1][1]), [x[1][0][0], x[1][0][1]]))
        ad_patients_w_background = ad_patients.leftOuterJoin(patient_stats) \
                                              .map(lambda x: x)
            #.map(lambda x: ((x[0], x[1][1]), (x[1][0][0], x[1][0][1])))

        nci = nci_patients_w_background.map(lambda x:
                                                    ((x[0], x[1][1]), x[1][0]))

        ad = ad_patients_w_background.map(lambda x: ((x[0], x[1][1]), x[1][0]))

        self.nci_reduced = nci.reduceByKey(lambda x, y: x + y)
        self.ad_reduced = ad.reduceByKey(lambda x, y: x + y)

top_k = Top_k()
rosmap_file = "s3n://ad.nci.patients/rosmap.csv"
gene_cluster_file = "s3n://ad.nci.patients/gene_cluster.csv"
start = time.time()
#print("start: ", start)
top_k.find_top_k(rosmap_file, gene_cluster_file)
top_k.print_top_k()
end = time.time()
#print("end: ", end)
print("Time: ", end - start)
