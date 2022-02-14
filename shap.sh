#!/bin/bash

for study_i in {1..2}
    do
    for model_i in {1..5}
        do 
        for i in {0..4}
            do 
               nohup python shap-in-parallel.py $study_i $model_i | tee -a output.txt &
            done
            sleep 10
        done
        sleep 3600
    done