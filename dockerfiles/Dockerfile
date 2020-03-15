# My dev environment for pyadjoint
#FROM salazardetroya/pyadjoint-firedrake
FROM pyadjoint:firedrake
MAINTAINER miguelito 

USER root
RUN apt-get -y update
RUN apt-get -y install software-properties-common 
RUN apt-get -y install silversearcher-ag
#RUN add-apt-repository ppa:neovim-ppa/stable
#RUN apt-get update
RUN apt-get -y install neovim curl git tmux

RUN git clone --depth 1 --quiet https://github.com/junegunn/fzf.git /home/firedrake/.fzf \
	 && /home/firedrake/.fzf/install --key-bindings --update-rc --completion

RUN curl -fLo /home/firedrake/.local/share/nvim/site/autoload/plug.vim --create-dirs --silent \
         https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

# Install nodejs for coc.nvim, -s -- -y to pass the options "-s --" and -y yes
# automatically
RUN curl -sL install-node.now.sh/lts | bash -s -- -y

USER firedrake
RUN echo "alias config='/usr/bin/git --git-dir=$HOME/.myconf/ --work-tree=$HOME'" >> $HOME/.bashrc
RUN echo "lol"
RUN echo ".myconf" >> .gitignore
RUN git clone --bare https://github.com/salazardetroya/.myconf.git $HOME/.myconf
RUN /usr/bin/git --git-dir=$HOME/.myconf/ --work-tree=$HOME checkout
RUN /usr/bin/git --git-dir=$HOME/.myconf/ --work-tree=$HOME config --local status.showUntrackedFiles no
RUN /usr/bin/git config --global user.email "salazardetro1@llnl.gov"
RUN /usr/bin/git config --global user.name "Miguel Salazar"

#USER root
#RUN find /home/firedrake -maxdepth 1 | sed "1d" | grep -v "/home/firedrake/shared" | xargs chown -R firedrake:firedrake 2> /dev/null || true
USER firedrake
RUN nvim -i NONE -c PlugInstall -c quitall #> /dev/null 2>&1
RUN nvim -i NONE -c 'CocInstall coc-python' -c 'sleep 2' -c quitall
ENV FZF_DEFAULT_COMMAND 'ag -g ""'

USER root


RUN wget http://gmsh.info/bin/Linux/gmsh-4.4.1-Linux64.tgz 
RUN tar -xvf gmsh-4.4.1-Linux64.tgz && \
	mv gmsh-4.4.1-Linux64/bin/* /usr/local/bin/ && \
	mv gmsh-4.4.1-Linux64/share/doc/ /usr/local/share/doc/ && \
	mv gmsh-4.4.1-Linux64/share/man/ /usr/local/share/man/
# It looks like for python3.6, the compiled ROL package uses RUNPATH instead of RPATH. Thus, when loading libteuchoscomm.so.12, which is compiled with RPATH being wherever it was compiled (likely a temporary folder), the library libteuchosparameterlist.so.12 is looked for in this directory and not in the one specified in the ROL package RUNPATH. Note that this does not occur for python3.5 where the compiled ROL package uses RPATH https://bitbucket.org/dolfin-adjoint/pyadjoint/issues/83/failed-to-import-rol-with-pyadjoint-201810
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.6/dist-packages/roltrilinos/lib

ENV VIRTUAL_ENV=/home/firedrake/firedrake
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install pudb meshio[all] pygmsh
#COPY pudb.cfg /home/fenics/.config/pudb/pudb.cfg
RUN echo "export PS1='\[\033[38;5;81m\]\u\[$(tput sgr0)\]\[\033[38;5;15m\]@\[$(tput sgr0)\]\[\033[38;5;85m\]docker\[$(tput sgr0)\]\[\033[38;5;15m\]:\[$(tput sgr0)\]\[\033[38;5;6m\][\[$(tput bold)\]\[$(tput sgr0)\]\[\033[38;5;225m\]\w\[$(tput sgr0)\]\[$(tput sgr0)\]\[\033[38;5;6m\]]$\[$(tput sgr0)\]\[\033[38;5;15m\]\[$(tput sgr0)\] '" >> /home/firedrake/.bashrc
RUN echo "alias tmux='TERM=xterm-256color tmux'" >> /home/firedrake/.bashrc
RUN echo "alias gst='git status'" >> /home/firedrake/.bashrc
RUN echo "alias gdh='git diff HEAD'" >> /home/firedrake/.bashrc

RUN pip3 install roltrilinos 
RUN pip3 install ROL
RUN pip3 install tensorflow==1.15.0
RUN pip3 install protobuf==3.7.1
RUN pip3 install --upgrade numpy

RUN apt-get -y install build-essential pkg-config python-dev python-six cython python-numpy coinor-libipopt1v5 coinor-libipopt-dev
RUN /bin/bash -l -c "source /home/firedrake/firedrake/bin/activate && GIT_SSL_NO_VERIFY=true pip3 install git+https://github.com/matthias-k/cyipopt.git"

RUN apt-get -y install language-pack-en-base

USER firedrake

WORKDIR /home/firedrake
