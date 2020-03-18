FROM firedrakeproject/firedrake
MAINTAINER miguelito 
USER root

RUN apt-get update && apt-get install -y wget libglu1 libxcursor-dev libxinerama1
RUN wget http://gmsh.info/bin/Linux/gmsh-4.4.1-Linux64.tgz 
RUN tar -xvf gmsh-4.4.1-Linux64.tgz && \
	mv gmsh-4.4.1-Linux64/bin/* /usr/local/bin/ && \
	mv gmsh-4.4.1-Linux64/share/doc/ /usr/local/share/doc/ && \
	mv gmsh-4.4.1-Linux64/share/man/ /usr/local/share/man/
RUN pip3 install meshio[all] pygmsh
USER firedrake
