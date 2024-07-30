#!/bin/bash

python3 ./get_layer_result.py --device CPU --infile ../results/csv/inconsistency.csv

python3 ./get_layer_result.py --device GPU --infile ../results/csv/inconsistency.csv
