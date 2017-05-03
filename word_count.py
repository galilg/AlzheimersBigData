import sys

from pyspark import SparkContext, SparkConf

if __name__ == "__main__":

    # create Spark context with Spark configuration
    conf = SparkConf().setAppName("Spark Count")
    sc = SparkContext(conf=conf)

    # get threshold
    threshold = int(sys.argv[1])

    # read in text file and split each document into words
    tokenized = sc.textFile("data2.csv").flatMap(lambda line: line.split("\n"))

    # count the occurrence of each word
    wordCounts = tokenized.map(lambda word: (word, 1)).reduceByKey(lambda v1,v2:v1 +v2)

    # filter out words with fewer than threshold occurrences
    filtered = wordCounts.filter(lambda pair:pair[1] >= threshold)

    # count characters
    charCounts = filtered.flatMap(lambda pair:pair[0]).map(lambda c: c).map(lambda c: (c, 1)).reduceByKey(lambda v1,v2:v1 +v2)

    char_list = charCounts.collect()
    word_list = filtered.collect()
    print("The character list is: {}\n".format(char_list))
    print("The word list is: {}".format(word_list))
