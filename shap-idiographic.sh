#!/bin/bash

for study_i in {3..5}
    do
    for model_i in {0..247}
        do
        for i in {0..9}
            do 
               nohup python shap-in-parallel-idiographic.py $study_i $model_i | tee -a output.txt &
            done
            sleep 10
        done
    done