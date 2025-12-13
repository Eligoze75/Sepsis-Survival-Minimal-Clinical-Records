# Addressing Feedback from Peer Review: DSCI 522: Group15

# Feedback 1 : License

"The license file should mention the names of every person in the group and 
the Creative Commons license should be used for the project report"" & 
"License: Since this project contains a report, the license file should have a creative commons license. 
The MIT license only covers the code within this repo. Also it should contain names of all team members. 
We missed that and lost points in milestone 1."

Changes: We have included the names of people in the group and creative commons license now. 
The link to the PR is here: https://github.com/Eligoze75/Sepsis-Survival-Minimal-Clinical-Records/pull/37


# Feedback 2 : License

"It might be helpful to describe a little more how the logistic model works, 
in particular how the coefficients were obtained and which hyper parameters were used."

Changes: We have included an additional section of how logistic regression model works. The PR link that addressed this change is here:
https://github.com/Eligoze75/Sepsis-Survival-Minimal-Clinical-Records/pull/45

Changes: 

# Feedback 3 : Docker Compose & Reproducibility

"The data analysis workflow isn't reproducible on my computer. The docker image seems to be jupyterlab according to the readme file, but the docker-compose.yml file is
* $ cat docker-compose.yml
* services:
*   analysis:
*     image: python:3.12
*     working_dir: /app
*     volumes:
*       - .:/app
*     command: bash
* 
* I got the following output when I ran docker compose up
* (base) harrisonlee@dhcp-128-189-57-117 ~/Code/ubc-mds/Block 3/DSCI522/peer-review/Sepsis-Survival-Minimal-Clinical-Records (main)
* $ docker compose up
* Attaching to analysis-1
* analysis-1 exited with code 0
"

Changes: We have updated all the dependencies, environment.yml, conda lock files, and docker-compose.yml  in this PR and merged it to the main branch already. 
All dependencies such as ipykernel, pandera, and pytest which were missing have already been added.

The commit that added docker-compose.yml is here: https://github.com/Eligoze75/Sepsis-Survival-Minimal-Clinical-Records/commit/4cce48b92886a50b5f680b6f4c509f059e2b3c72

The PR link for creating environment.yml, conda lock files, and docker-compose.yml is here:
https://github.com/Eligoze75/Sepsis-Survival-Minimal-Clinical-Records/pull/40



# Feedback 4 : 

"4. I would recommend including a brief description of the figure inside the caption describing the axis of the plots and label the subplots 
(eg. 1a, 1b, 1c for figure 1) for clarity. So the reader can tell what the plot is trying to convey without reading the entire paper.
Using figure 1a for example: Figure 1a: The distribution of survival outcomes across age is shown above, with age on the x-axis and the counts on the y-axis, 
grouped by survival outcomes..."


Changes: We have changed the figrue caption accordingly. The PR link that addressed this change is here:
https://github.com/Eligoze75/Sepsis-Survival-Minimal-Clinical-Records/pull/45


# Feedback 5 :

"5. I found a minor formatting issues in the report - [SHAP (SHapley Additive exPlanations)]"

Changes: We have fixed this typo here. The PR link that addressed this change is here:
https://github.com/Eligoze75/Sepsis-Survival-Minimal-Clinical-Records/pull/45
