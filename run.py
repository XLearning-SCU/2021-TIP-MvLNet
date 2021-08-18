import sys, os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data import get_data
from applications.MvLNet import run_net
from applications.Config import load_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = load_config("./config/noisymnist.yaml")  # noisymnist　Caltech101-20 wiki
# LOAD DATA
data_list = get_data(config)

# RUN EXPERIMENT
x_final_list, scores = run_net(data_list, config)

   

