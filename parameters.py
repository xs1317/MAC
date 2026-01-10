import argparse


DATA_FOLDER = "./data"
parser = argparse.ArgumentParser()


# model config
parser.add_argument("--model_name", help="model name", type=str)
parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
parser.add_argument("--dataset", help="name of the dataset", type=str, default='mit-states')
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--epoch_start", help="start epoch", default=0, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=48, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=16, type=int)
parser.add_argument("--num_workers", help="number of workers", default=10, type=int)
parser.add_argument("--context_length", help="sets the context length of the clip model", default=8, type=int)
parser.add_argument("--attr_dropout", help="add dropout to attributes", type=float, default=0.3)
parser.add_argument("--yml_path", help="yml path", type=str)
parser.add_argument("--dataset_path", help="dataset path", type=str)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n", default=5, type=int, help="saves the model every n epochs")
parser.add_argument("--save_final_model", help="indicate if you want to save the model state dict()", action="store_true")
parser.add_argument("--load_model", default=None, help="load the trained model")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=int)
parser.add_argument("--same_prim_sample", help="if sample same prim samples", action="store_true")

parser.add_argument("--open_world", help="evaluate on open world setup", default=False)
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)
parser.add_argument("--text_encoder_batch_size", help="batch size of the text encoder", default=16, type=int)
parser.add_argument('--threshold', type=float, default=None, help="optional threshold")
parser.add_argument('--threshold_trials', type=int, default=50, help="how many threshold values to try")

parser.add_argument("--adapter_dim", help="middle dimension of Adapter", type=int, default=64)
parser.add_argument("--init_lamda", help="lamda initialization value", type=float, default=0.1)
parser.add_argument("--cmt_layers", help="Number of layers in cross-attention", type=int, default=2)


parser.add_argument("--wandb", help="Number of layers in cross-attention", type=bool, default=True)
parser.add_argument("--evalute_train_set", type=bool,  default=False)
parser.add_argument("--evalute_test_set", type=bool,  default=True)


parser.add_argument("--mit_states_path",help="path for mit_states",default="/root/autodl-tmp/data/CZSL/mit-states")
parser.add_argument("--cgqa_path",help="path for cgqa",default="/root/autodl-tmp/data/CZSL/cgqa")


parser.add_argument("--loss_fn",default="multi")
parser.add_argument("--debug",default=False)




# for GCE


# Model parameters
parser.add_argument('--model', default='graphfull', help='visprodNN|redwine|labelembed+|attributeop|tmn|compcos')
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of share embedding space')
parser.add_argument('--nlayers', type=int, default=3, help='Layers in the image embedder')
parser.add_argument('--nmods', type=int, default=24, help='number of mods per layer for TMN')
parser.add_argument('--embed_rank', type=int, default=64, help='intermediate dimension in the gating model for TMN')
parser.add_argument('--update_features', action = 'store_true', default=False, help='If specified, train feature extractor')
parser.add_argument('--freeze_features', action = 'store_true', default=False, help='If specified, put extractor in eval mode')
parser.add_argument('--emb_init', default=None, help='w2v|ft|gl|glove|word2vec|fasttext, name of embeddings to use for initializing the primitives')
parser.add_argument('--clf_init', action='store_true', default=False, help='initialize inputs with SVM weights')
parser.add_argument('--static_inp', action='store_true', default=False, help='do not optimize primitives representations')
parser.add_argument('--composition', default='mlp_add', help='add|mul|mlp|mlp_add, how to compose primitives')
parser.add_argument('--relu', action='store_true', default=False, help='Use relu in image embedder')
parser.add_argument('--dropout', action='store_true', default=False, help='Use dropout in image embedder')
parser.add_argument('--norm', action='store_true', default=False, help='Use normalization in image embedder')
parser.add_argument('--train_only', action='store_true', default=False, help='Optimize only for train pairs')

#CGE
parser.add_argument('--graph', action='store_true', default=False, help='graph l2 distance triplet') # Do we need this
parser.add_argument('--graph_init', default=None, help='filename, file from which initializing the nodes and adjacency matrix of the graph')
parser.add_argument('--gcn_type', default='gcn', help='GCN Version')

# Forward
parser.add_argument('--eval_type', default='dist', help='dist|prod|direct, function for computing the predictions')

# Primitive-based loss
parser.add_argument('--lambda_aux', type=float, default=0.0, help='weight of auxiliar losses on the primitives')


# Hyperparameters
parser.add_argument("--fc_emb", default='768,1024,1200',help="Image embedder layer config")
parser.add_argument("--gr_emb", default='d4096,d',help="graph layers config")



# For Cross Model Test
