from train import ModelSelector
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import copy 
from networkx.algorithms.components.connected import connected_components
import json
import logging
import time

def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current    

def get_often_confused_bricks(df, init_threshold=100, confusion_threshold=3):
    sus_items = []
    sus_dict = {}
    for item in df_cm:
        correct = df_cm[item][item]
        if correct < init_threshold:
            sus_items.append(item)
    for item in sus_items:
        sus_dict[item] = dict(df_cm[item].loc[df_cm[item]>confusion_threshold])
    return sus_dict

def get_parts_to_join(confused_bricks):
    parts_to_join = []
    for key, value in confused_bricks.items():
        mistaken_parts = list(value.keys())
        for part in mistaken_parts:
            if part == key:
                continue
            if part in confused_bricks.keys():
                if key in confused_bricks[part]:
                    if not [part,key] in parts_to_join and not [key, part] in parts_to_join:
                        parts_to_join.append([key,part])
    return parts_to_join

def get_reorg_dict(l):
    reorg_dict = {}
    for i in l:
        reorg_key = '_'.join(i)
        reorg_dict[reorg_key] = list(i)
    return reorg_dict

def get_bidirectional_scheme(df, threshold=5):
    sus_bricks = get_often_confused_bricks(df, confusion_threshold=threshold)
    parts_to_join = get_parts_to_join(sus_bricks)
    G = to_graph(parts_to_join)
    bidirectional_chains = list(connected_components(G))
    return get_reorg_dict(bidirectional_chains)

def get_unidirectional_scheme(df, threshold=5):
    new_arr = copy.copy(df.values)
    new_arr = np.where(new_arr > threshold, new_arr, 0)
    class_names = list(df.index)
    graph = nx.from_numpy_matrix(new_arr, create_using=nx.Graph)
    unidirectional_chains = []
    for elem in connected_components(graph):
        if len(elem) != 1:
            unidirectional_chains.append(list(map(lambda x: class_names[x], elem)))
    return get_reorg_dict(unidirectional_chains)


if __name__ == '__main__':
  #  logging.basicConfig(level=logging.INFO)
  #  parser = argparse.ArgumentParser()
  #  parser.add_argument('-i', required=True,
   #                     help='config json file name', dest='json')
  #  args = parser.parse_args()
    default_dict = {
        'model_path': '...',
        'wandb_run_name': 'kzawora/lego-4h/runs/66z3b5tl',
        'test_set_path': '/macierz/home/s165115/legosymlink/kzawora/dataset_new_more_sizes_3/resized_224x224/val',
        'model': 'VGG16',
        'img_width': 224,
        'img_height': 224,
        'batch_size': 128,
        'evaluate': True,
        'dump_report': True,
        'class_names':  ['10197', '10201', '10314', '10928', '11090', '11153', '11211', '11212', '11213', '11214', '11215', '11272', '11458', '11476', '11477', '11478', '120493', '131673', '13349', '13547', '13548', '13731', '14395', '14417', '14419', '14704', '14716', '14720', '14769', '15068', '15070', '15092', '15100', '15254', '15332', '15379', '15395', '15397', '15460', '15461', '15470', '15535', '15573', '15672', '15706', '15712', '158788', '15967', '16577', '17114', '17485', '18649', '18651', '18653', '18674', '18838', '18969', '19159', '20896', '21229', '216731', '22385', '22388', '22390', '22391', '22885', '22888', '22889', '22890', '22961', '2357', '239356', '24014', '24122', '2412b', '2419', '2420', '242434', '24246', '24299', '24316', '24375', '2441', '2445', '2450', '24505', '2453b', '2454', '2456', '2460', '2476a', '2486', '24866', '2540', '254579', '26047', '2639', '2654', '26601', '26604', '267165', '27255', '27262', '27266', '2730', '2736', '274829', '27940', '2853', '2854', '28653', '2877', '2904', '2926', '292629', '296435', '30000', '3001', '3002', '3003', '3004', '30044', '30046', '3005', '30069', '3008', '3009', '30099', '3010', '30136', '30157', '30165', '3020', '3021', '3022', '30237b', '3024', '3028', '3031', '3032', '3033', '3034', '3035', '30357', '30361c', '30363', '30367c', '3037', '3038', '30387', '3040b', '30414', '3045', '30503', '30552', '30553', '30565', '3062', '3068', '3069b', '3070b', '3185', '32000', '32002', '32013', '32016', '32017', '32028', '32034', '32039', '32054', '32056', '32059', '32062', '32064a', '32073', '32123b', '32124', '32140', '32184', '32187', '32192', '32198', '32250', '32291', '32316', '32348', '3245', '32526', '32529', '32530', '32557', '32828', '32932', '3298', '33291', '33299b', '33909', '3460', '35044', '3622', '3623', '3633', '3639', '3640', '3659', '3660', '3665', '3666', '3673', '3676', '3679', '3680', '36840', '36841', '3700', '3701', '3705', '3706', '3707', '3710', '3713', '374125', '3747b', '3795', '3832', '3895', '392043', '3941', '3942c', '3957', '3958', '39739', '4032a', '40490', '4073', '4081b', '4083', '40902', '413097', '41530', '41532', '4162', '41677', '41678', '41682', '41740', '41747', '41748', '41762', '41768', '41769', '41770', '4185', '42003', '42023', '4216', '4218', '42446', '4274', '4282', '4286', '4287b', '43708', '43712', '43713', '43898', '44126', '44568', '4460b', '44728', '4477', '44809', '44861', '4488', '4490', '4510', '4519', '45590', '456218', '45677', '4600', '465007', '4727', '4733', '47397', '47398', '4740', '4742', '47456', '474589', '47753', '47755', '47905', '48092', '48171', '48336', '4865b', '4871', '48723', '48729b', '48933', '48989', '496432', '49668', '50304', '50305', '50373', '50950', '51739', '52031', '523081', '52501', '53899', '54383', '54384', '55013', '551028', '56596', '569005', '57519', '57520', '57585', '57895', '57909b', '58090', '59426', '59443', '60032', '6005', '6014', '6020', '60470b', '60471', '60474', '60475b', '60476', '60477', '60478', '60479', '60481', '60483', '60484', '60581', '60592', '60593', '60594', '60596', '60598', '60599', '60607', '60608', '60616b', '60621', '60623', '608036', '6081', '60897', '6091', '61070', '61071', '6111', '61252', '612598', '61409', '614655', '61484', '6157', '6182', '6215', '6222', '6231', '6232', '6233', '62462', '63864', '63868', '64225', '64288', '64391', '64393', '64681', '64683', '64712', '6536', '6541', '6553', '6558', '6587', '6628', '6632', '6636', '72454', '74261', '77206', '822931', '84954', '85080', '852929', '853045', '85984', '87079', '87081', '87083', '87087', '87544', '87580', '87609', '87620', '87697', '88292', '88323', '88646', '88930', '901078', '90195', '90202', '90609', '90611', '90630', '915460', '92013', '92092', '92582', '92583', '92589', '92907', '92947', '92950', '93273', '93274', '93606', '94161', '959666', '966967', '98100', '98197', '98262', '98283', '98560', '99008', '99021', '99206', '99207', '99773', '99780', '99781'],
        'get_reorg_schemes': True
    }

    init_dict = copy.deepcopy(default_dict)
#    with open(str(args.json)) as json_file:
#        args = json.load(json_file)
#        init_dict.update(args)

    best_model = wandb.restore(
        'model-best.h5', run_path=init_dict['wandb_run_name'])
    ms = ModelSelector(init_dict, init_build=False)
    test_datagen = ImageDataGenerator(
        preprocessing_function=ms.get_preprocessing_function())
    test_generator = test_datagen.flow_from_directory(init_dict['test_set_path'],
                                                      target_size=(
                                                          init_dict['img_width'], init_dict['img_height']),
                                                      batch_size=init_dict['batch_size'], shuffle=False)
    model = tf.keras.models.load_model(best_model.name)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if default_dict['evaluate']:
        eval_results = model.evaluate(test_generator, verbose=1)
    preds = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    class_names = init_dict['class_names']
    if init_dict['dump_report']:
        report = classification_report(test_generator.classes, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'classification_report_{init_dict["model"]}_{timestr}.csv')
    conf_matrix = confusion_matrix(test_generator.classes, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    mul=12
    plt.figure(figsize = (10*mul, 7*mul), dpi=40)
    sns_plot = sn.heatmap(df_cm, square=True, cmap='viridis')
    sns_plot.figure.savefig('confusion_matrix.png')
    
    if init_dict['get_reorg_schemes']:
        bidirectional = get_bidirectional_scheme(df_cm)
        unidirectional = get_unidirectional_scheme(df_cm)

        with open(f'bidirectional_reorg_{init_dict["model"]}_{timestr}.json', 'w') as fp:
            json.dump(bidirectional, fp)

        with open(f'unidirectional_reorg_{init_dict["model"]}_{timestr}.json', 'w') as fp:
            json.dump(unidirectional, fp)