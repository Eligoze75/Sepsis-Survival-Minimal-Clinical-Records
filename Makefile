.PHONY : all report clean

all: 01_data_loading \
	02_data_transformation \
	03_run_eda \
	04_modeling_and_evaluation \
	report

01_data_loading : src/01_data_loading.py
	python src/01_data_loading.py

02_data_transformation : src/02_data_transformation.py
	python src/02_data_transformation.py

03_run_eda : src/03_run_eda.py
	python src/03_run_eda.py

04_modeling_and_evaluation : src/04_modeling_and_evaluation.py
	python src/04_modeling_and_evaluation.py


report :
	quarto render reports/sepsis-predictor-report.qmd

clean :
	rm -f results/figure/isles.png \
		results/figure/abyss.png \
		results/figure/last.png \
		results/figure/sierra.png \
		results/isles.dat \
		results/abyss.dat \
		results/last.dat \
		results/sierra.dat