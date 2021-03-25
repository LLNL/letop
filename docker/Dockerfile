FROM firedrakeproject/firedrake
MAINTAINER miguelito 
USER root

RUN apt-get update && apt-get install -y wget libglu1 libxcursor-dev libxinerama1
ARG gmsh_version=4.7.1
RUN wget http://gmsh.info/bin/Linux/gmsh-$gmsh_version-Linux64.tgz
RUN tar -xvf gmsh-$gmsh_version-Linux64.tgz && \
	mv gmsh-$gmsh_version-Linux64/bin/* /usr/local/bin/ && \
	mv gmsh-$gmsh_version-Linux64/share/doc/ /usr/local/share/doc/ && \
	mv gmsh-$gmsh_version-Linux64/share/man/ /usr/local/share/man/
RUN pip3 install meshio[all] pygmsh
USER firedrake
