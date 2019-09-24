####################################################################
# Original Image
####################################################################

# This image can be changed to a simpler one:
FROM python:3.6

####################################################################
# Documentation
####################################################################

# You can change this:
LABEL author = "philippefanaro@gmail.com"

# When building:

# sudo docker build -t lifetimes .

# To run it with a volume (example):

# sudo docker run -v /home/<user_name>/lifetimes/:/usr/share/local_lifetimes -p 80:80 -it lifetimes

####################################################################
# Setup
####################################################################

RUN pip install -I scipy==1.1.0                                 && \
    pip install matplotlib                                      && \
    pip install seaborn                                         && \
    # pip install lifetimes                                       && \
    # Sphinx Docs (Optional):
    pip install sphinx-rtd-theme                                && \                                  && \
    pip install sphinxcontrib-napoleon                          && \
    pip install recommonmark                                    && \
    pip install sphinxcontrib-bibtex                            

####################################################################
# Adding Source Code to the Docker
####################################################################

# Para quando estiver em produção e não precisar mais montar volume:

# COPY . /lifetimes

####################################################################
# Initial Commands
####################################################################

# `pip install -e .` installs your local version of the library
# overriding the initial installation

CMD cd /usr/share/local_lifetimes                               && \
    pip install -e .                                            && \
    /bin/bash                             