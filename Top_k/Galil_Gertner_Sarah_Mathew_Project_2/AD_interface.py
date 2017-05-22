#!/bin/python3

#---- Imports -----------------------------------------------------------------

import pprint
import time

from MachineLearning import MachineLearning
from Top_k import Top_k as tk

#---- Menu --------------------------------------------------------------------

def load_top_ten():
    print("Please load top 10 first (command 1).")


def main_menu():
    # Instantiate the Top_k class
    top_k = tk()

    spark_run = False
    command = 0
    while command is not 'Q' and command is not 'q':
        command = input("""

-------------------------------------------------------------------------------
|                                                                             |
|                                                                             |
|                                                                             |
|                                                                             |
|                                                                             |
|                                                                             |
|                            Alzheimer's -NCI                                 |
|                                 Top-K                                       |
|                                                                             |
|                                                                             |
|                                                                             |
| 1 - Print top 10 t-test values                                              |
| 2 - Print cluster mean and std                                              |
| 3 - Run machine learning                                                    |
| Q - Exit                                                                    |
-------------------------------------------------------------------------------

Enter Command >>> """)
        #command = input()

        if command == '1':
            rosmap_file = input("Enter name of ROSMAP file: ")
            gene_cluster_file = input("Enter name of gene cluster file: ")
            start = time.time()
            top_k.find_top_k(rosmap_file, gene_cluster_file)
            top_k.print_top_k()
            end = time.time()
            print("Time: ", end - start)
            spark_run = True

        elif command == '2':
            if spark_run:
                cluster = input("Cluster number: ")
                top_k.print_cluster_values(cluster)
            else:
                load_top_ten()

        elif command == '3':
            if spark_run:
                top_k.top_k_vals_for_patients()
                ad, nci = top_k.get_all_ad_and_nci_vals()
                sc = top_k.get_spark_context()
                #import pdb; pdb.set_trace()

                # Instantiate the MachineLearning class
                ml = MachineLearning(ad, nci, sc)

                # Create the labeled points for models with both
                # life stats and no life stats
                ml.set_labeled_points_data()

                model_type = input("Include age stats? (y/n): ")
                if model_type == 'y' or model_type == 'Y':
                    group = 'life_stats'
                else:
                    group = 'genes'

                num_rf_trees = input("Enter number of trees in random forest: ")

                # Build model
                ml.build_model(group, num_rf_trees)
                ml.print_all_data()
            else:
                load_top_ten()

    print("Bye")


main_menu()
