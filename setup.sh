# create conda env with local R
conda create --name ts1
conda activate ts1
conda install rpy2

# install R packages
R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'

# if there are any problems with installing R packages due to curl (dependency of forecast):
# apt update
# apt upgrade
# apt-get install libcurl4-openssl-dev libxml2-dev
# apt-get install curl libssl-dev
# reboot
# R
# install.packages("curl")
# Configuration failed because libcurl was not found? Install manually!
# use the temporarily downloaded <curl-file.gz> from the error message!
# find out where `libcurl.pc` is (perhaps `/usr/lib/x86_64-linux-gnu/pkgconfig` ?)
# R CMD INSTALL --configure-vars='LIB_DIR=<libcurl.pc directory>' <curl-file.gz>

# install additional packages
pip install "gluonts[mxnet,pro]"
pip install pynvml
# manually clone and install rapl