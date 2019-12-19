from continuumio/miniconda3

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install make && conda update -n base -c defaults conda

# Use the environment.yml to create the conda environment.
ADD environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN [ "conda", "env", "create" ]

ADD . /afq-insight-paper
WORKDIR /afq-insight-paper

# Set the ENTRYPOINT to use bash
RUN [ "/bin/bash", "-c", "echo 'conda activate afq-insight-paper' >> /root/.bashrc" ]
ENTRYPOINT [ "/bin/bash", "-c" ]
CMD [ "source activate afq-insight-paper && jupyter lab --ip 0.0.0.0 --port 8899 --no-browser --allow-root ." ]

