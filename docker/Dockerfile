FROM firedrakeproject/firedrake-env:latest

# This DockerFile is looked after by
MAINTAINER Miguel Salazar <salazardetroya@gmail.com>
USER root

RUN apt-get update && apt-get install -y wget libglu1 libxcursor-dev libxinerama1
RUN pip3 install meshio[all] pygmsh
USER firedrake
WORKDIR /home/firedrake

# Now install Firedrake.
RUN curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
RUN bash -c "python3 firedrake-install --no-package-manager --disable-ssh --remove-build-files --pip-install scipy --doi 10.5281/zenodo.5526481"
