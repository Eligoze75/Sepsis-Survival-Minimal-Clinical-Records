FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*
 
USER ${NB_USER}

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda create -n sepsis_survival_venv --file /tmp/conda-linux-64.lock --copy -y \
 && conda clean -afy
RUN fix-permissions "${CONDA_DIR}" \
 && fix-permissions "/home/${NB_USER}"

SHELL ["bash", "-lc"]
ENV CONDA_DEFAULT_ENV=sepsis_survival_venv
ENV PATH="${CONDA_DIR}/envs/${CONDA_DEFAULT_ENV}/bin:${PATH}"