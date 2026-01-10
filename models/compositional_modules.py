import os


from models.coop import coop
from models.csp import get_csp, get_mix_csp
from models.troika import get_troika
from models.tp_sa import get_tp_sa
from models.dfsp import get_dfsp
from models.CAILA.caila import get_caila
from models.tp_sa_hardC_disent import get_tp_sa_hardC_disent
from models.CGE.graph_method import get_gce

from models.CspModel.models.GraphPrompt import get_graph_and_prompt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device):
    if config.model_name == "coop":
        return coop(train_dataset, config, device)

    elif config.model_name == "csp":
        return get_csp(train_dataset, config, device)

    # special experimental setup
    elif config.model_name == "mix_csp":
        return get_mix_csp(train_dataset, config, device)
    elif config.model_name == "troika":
        return get_troika(train_dataset, config, device)
    elif config.model_name == "tp_sa":
        return get_tp_sa(train_dataset, config, device)
    elif config.model_name == "dfsp":
        return get_dfsp(train_dataset, config, device)
    elif config.model_name == "CAILA":
        return get_caila(train_dataset, config, device)
    elif config.model_name == "gce":
        return get_gce(train_dataset, config, device)
    elif config.model_name == "tp_sa_hardC_disent":
        return get_tp_sa_hardC_disent(train_dataset, config, device)
    elif config.model_name == "gipcol":
        return get_graph_and_prompt(train_dataset, config, device)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Experiment Name {:s}.".format(
                config.model_name
            )
        )
