#!/bin/python

#---- Imports -----------------------------------------------------------------


import pprint
import time

from Top_k import Top_k as tk

#---- Menu --------------------------------------------------------------------

def load_top_ten():
    print("Please load top 10 first (command 1).")


def main_menu():
    # Instantiate the class
    top_k = tk()
    spark_run = False
    command = 0
    while command is not 'Q' and command is not 'q':
        command = raw_input("""

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
| 3 - Print top two AD patients with life stats                               |
| Q - Exit                                                                    |
-------------------------------------------------------------------------------

Enter Command >>> """)
        #command = input()

        if command == '1':
            #rosmap_file = raw_input("Enter name of ROSMAP file: ")
            #gene_cluster_file = raw_input("Enter name of gene cluster file: ")
            rosmap_file = "rosmap.csv"
            gene_cluster_file = "gene_cluster.csv"
            start = time.time()
            top_k.find_top_k(rosmap_file, gene_cluster_file)
            top_k.print_top_k()
            end = time.time()
            print("Time: ", end - start)
            spark_run = True

        elif command == '2':
            if spark_run:
                cluster = raw_input("Cluster number: ")
                top_k.print_cluster_values(cluster)
            else:
                load_top_ten()

        elif command == '3':
            if spark_run:
                top_k.top_k_vals_for_patients()
                top_k.print_first_two_AD_patients_with_life_stats()
            else:
                load_top_ten()

    print("Bye")


main_menu()
