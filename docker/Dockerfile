FROM firedrakeproject/firedrake
MAINTAINER miguelito 
USER root

RUN apt-get update && apt-get install -y wget libglu1 libxcursor-dev libxinerama1
RUN pip3 install meshio[all] pygmsh
USER firedrake
