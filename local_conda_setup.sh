# conda config --set channel_priority strict
# conda install -n base conda-libmamba-solver
# conda config --set solver libmamba
# yes | conda clean --all

mamba create -n pigeon-pack
eval "$(mamba shell hook --shell $SHELL)"
mamba activate
echo "This can take a really long time. Go get some coffee."
sleep 5
yes | mamba env -n pigeon-pack update -v -f "environment.yaml"