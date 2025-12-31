.PHONY : all report clean \
	data_loading_train data_loading_test \
	data_transformation run_eda modeling_and_evaluation

all: data_loading_train \
	data_loading_test \
	data_transformation \
	run_eda \
	modeling_and_evaluation \
	report

data_loading_train : src/data_loading.py
	python src/data_loading.py

data_loading_test : src/data_loading.py
	python src/data_loading.py \
	--filename s41598-020-73558-3_sepsis_survival_study_cohort.csv

data_transformation : src/data_transformation.py
	python src/data_transformation.py

run_eda : src/run_eda.py
	python src/run_eda.py

modeling_and_evaluation : src/modeling_and_evaluation.py
	python src/modeling_and_evaluation.py

report :
	quarto render reports/sepsis-predictor-report.qmd

clean :
	rm -f results/figures/* \
		results/tables/*
