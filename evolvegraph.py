import osos.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'import config as cfgimport torchimport timeimport numpy as npfrom argparse import ArgumentParserfrom instructors.train import trainingfrom utils.general import read_pickle, pool_mapfrom model.models import EvolveGraphfrom model.encoder import static_graph_learning, dynamic_graph_learningfrom model.decoder import states_predictfrom model.context import context_embeddingfrom torch.nn.parallel import DataParallelfrom itertools import permutationsfrom numpy.random import randomfrom tqdm import trangedef init_args():    parser = ArgumentParser()    parser.add_argument('--data_type', type=str, default='NBA')    parser.add_argument('--input_dim', type=int, default=2)    parser.add_argument('--cuda', type=int, default=2)    # parser.add_argument('--agent_size', type=int, default=5)    # parser.add_argument('--suffix', action='store_true', default=False)    return parser.parse_args()def load_data(args):    # suffix = '_' if args.suffix else ''    path = 'data/{}/data_100seg.pkl'.format(args.data_type)    train, val, test = read_pickle(path)    #data = read_pickle(path)    #data = [[i[:5000] for i in j] for j in data]    #train, val, test = data    data = {'train': train, 'val': val, 'test': test}    return datadef run():    args = init_args()    cfg.init_args(args)    data = load_data(args)    encoder = dynamic_graph_learning(args.input_dim, cfg.context_emb_hid, cfg.edge_types)    decoder = states_predict(args.input_dim, cfg.context_emb_hid, cfg.edge_types)    cont_emb = context_embedding(cfg.context_dim, cfg.context_emb_hid)    model = EvolveGraph(encoder, decoder, cont_emb)    model = DataParallel(model)    if cfg.using_gpu:        model = model.cuda()    ins = training(model, data, args)    ins.train()if __name__ == "__main__":    run()