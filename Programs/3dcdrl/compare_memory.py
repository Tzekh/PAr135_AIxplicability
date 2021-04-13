import ujson
from random import randint

import numpy as np
import torch
from torch.autograd import Variable

from arguments import parse_game_args
from doom_evaluation import BaseAgent
from environments import DoomEnvironment
from models import CNNPolicy
import base64
import io
from PIL import Image


def gen_classic(selh, file, scenario = False, model="model_final"):
    params = parse_game_args()


    # Charge le scénario
    if not scenario :
        params.scenario = "custom_scenario003.cfg"
    else:
        params.scenario = scenario

    env = DoomEnvironment(params)

    device = torch.device("cuda" if False else "cpu")

    num_actions = env.num_actions
    network = CNNPolicy(3, num_actions, True, (3, 64, 112)).to(device)

    # Chargement du modèle de base

    network = CNNPolicy(3, num_actions, True, (3, 64, 112)).to(device)

    checkpoint = torch.load('models/' + model + '.pth.tar', map_location=lambda storage, loc: storage)

    """Remplacement des clefs du dictionnaire qui posent problème"""

    checkpoint['model']["dist.linear.weight"] = checkpoint['model']["dist_linear.weight"]
    del checkpoint['model']["dist_linear.weight"]
    checkpoint['model']["dist.linear.bias"] = checkpoint['model']["dist_linear.bias"]
    del checkpoint['model']["dist_linear.bias"]

    network.load_state_dict(checkpoint['model'])

    agent = BaseAgent(network, params)

    ERU = {'env': env, 'agent': agent}

    # Chargement des checkpoints
    num_checkpoints = [98, 98, 159]
    checkpoints = [1]*sum(num_checkpoints)
    networks = [1]*sum(num_checkpoints)
    agents = [1]*sum(num_checkpoints)
    ERUs = [1]*sum(num_checkpoints)

    for i in range(len(num_checkpoints)):
        for j in range(num_checkpoints[i]):
            iter = i*num_checkpoints[0]+j

           # if i==0:
           #     checkpoint_filename = '/home/adam/Bureau/Transfer Learning/5 - 28-03-21/checkpoint_{}_{}.pth.tar'.format(str(i + 1), str(j + 88))
            #else:
            checkpoint_filename = '/home/adam/Bureau/Transfer Learning/5 - 28-03-21/checkpoint_{}_{}.pth.tar'.format(str(i + 1), str(j + 1))

            checkpoints[i*num_checkpoints[0]+j] = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage)

            """Remplacement des clefs du dictionnaire qui posent problème"""

            checkpoints[iter]['model']["dist.linear.weight"] = checkpoints[iter]['model']["dist_linear.weight"]
            del checkpoints[iter]['model']["dist_linear.weight"]
            checkpoints[iter]['model']["dist.linear.bias"] = checkpoints[iter]['model']["dist_linear.bias"]
            del checkpoints[iter]['model']["dist_linear.bias"]

            networks[iter] = CNNPolicy(3, num_actions, True, (3, 64, 112)).to(device)
            networks[iter].load_state_dict(checkpoints[iter]['model'])

            agents[iter] = BaseAgent(networks[iter], params)

            ERUs[iter] = {'env': env, 'agent': agents[iter]}

            ERUs[iter]['env'].reset()

    selhs = []
    for i in range(sum(num_checkpoints)):
        selh = tsne_1d_projection(127)
        selh = torch.from_numpy(selh).type(torch.FloatTensor)
        selh = Variable(selh, volatile=True)
        selhs.append(selh)


    scores = []
    hiddens = []
    inputs = []
    actions = []

    #Boucle pour obtenir les images du modèle de base

    obss = []
    actions = []

    for i in range(50):
        obs = ERU['env'].get_observation()
        action, value, action_probs, grads = ERU['agent'].get_action_value_and_probs_zeroes(obs, selh, epsilon=0.0)
        ERU['env'].make_action(int(action))
        obss.append(obs)
        actions.append(action)


    #Boucle pour évaluer les checkpoints sur les situations du modèle de base

    for i in range(sum(num_checkpoints)):

        for obs2 in obss:
            action, value, action_probs, grads = ERUs[i]['agent'].get_action_value_and_probs_zeroes(obs2, selhs[i], epsilon=0.0)

        hidden = ERUs[i]['agent'].model.get_gru_h()
        h = ''
        for elem in hidden[0][0]:
            h += str(elem) + ","
        h = h[:-1]

        h = h.split(',')
        hiddens.append(h)

        ERU['env'].make_action(int(action))

    im = Image.new('P', (sum(num_checkpoints), 128))
    for i in range(len(hiddens)):
        for j in range(len(hiddens[i])):
            value = int((float(hiddens[i][j])+1)*255/2)
            im.putpixel((i, j), (value, value, value, 255))
    im.show()
    im.save("timeline.png")

    im = Image.new('P', (sum(num_checkpoints)-1, 128))
    for i in range(len(hiddens)-1):
        for j in range(len(hiddens[i])):
            value = int((abs(float(hiddens[i][j])-float(hiddens[i+1][j])))*255*1.5)
            if value>255:
                value=255
            im.putpixel((i, j), (value, value, value, 255))
    im.show()
    im.save("variation.png")


def remove_all():
    return np.full(
        shape=128,
        fill_value=0.02,
        dtype=np.float)


def top(n):
    top = [92, 12, 105, 13, 75, 74, 93, 2, 108, 85, 115, 21, 72, 66, 65, 56, 11, 30, 116, 79, 106, 87, 86, 31, 17, 18, 100, 120, 23, 67, 27, 101, 50, 62, 34, 113, 78, 10, 7, 127, 8, 69, 119, 39, 49, 52, 104, 94, 111, 112, 84, 122, 28, 46, 15, 32, 16, 103, 19, 98, 125, 25, 107, 70, 0, 60, 37, 68, 117, 41, 35, 61, 89, 53, 44, 20, 102, 126, 114, 82, 77, 124, 5, 51, 55, 80, 6, 40, 22, 99, 97, 73, 91, 76, 36, 110, 90, 14, 38, 59, 54, 88, 83, 118, 45, 58, 121, 47, 9, 63, 95, 33, 109, 1, 71, 57, 29, 43, 64, 81, 26, 42, 4, 123, 96, 3, 48, 24]

    apply_oder(n, top)


def change(n):
    ch = [92, 12, 105, 13, 75, 74, 93, 2, 108, 85, 115, 21, 72, 66, 65, 56, 11, 30, 116, 79, 106, 87, 86, 31, 17, 18, 100, 120, 23, 67, 27, 101, 50, 62, 34, 113, 78, 10, 7, 127, 8, 69, 119, 39, 49, 52, 104, 94, 111, 112, 84, 122, 28, 46, 15, 32, 16, 103, 19, 98, 125, 25, 107, 70, 0, 60, 37, 68, 117, 41, 35, 61, 89, 53, 44, 20, 102, 126, 114, 82, 77, 124, 5, 51, 55, 80, 6, 40, 22, 99, 97, 73, 91, 76, 36, 110, 90, 14, 38, 59, 54, 88, 83, 118, 45, 58, 121, 47, 9, 63, 95, 33, 109, 1, 71, 57, 29, 43, 64, 81, 26, 42, 4, 123, 96, 3, 48, 24]
    apply_oder(n, ch)


def tsne_1d_projection(n):
    #proj = [92, 12, 105, 13, 75, 74, 93, 2, 108, 85, 115, 21, 72, 66, 65, 56, 11, 30, 116, 79, 106, 87, 86, 31, 17, 18, 100, 120, 23, 67, 27, 101, 50, 62, 34, 113, 78, 10, 7, 127, 8, 69, 119, 39, 49, 52, 104, 94, 111, 112, 84, 122, 28, 46, 15, 32, 16, 103, 19, 98, 125, 25, 107, 70, 0, 60, 37, 68, 117, 41, 35, 61, 89, 53, 44, 20, 102, 126, 114, 82, 77, 124, 5, 51, 55, 80, 6, 40, 22, 99, 97, 73, 91, 76, 36, 110, 90, 14, 38, 59, 54, 88, 83, 118, 45, 58, 121, 47, 9, 63, 95, 33, 109, 1, 71, 57, 29, 43, 64, 81, 26, 42, 4, 123, 96, 3, 48, 24]
    proj = [i for i in range(128)]
    return apply_oder(n, proj)


def apply_oder(n, order):
    assert n < 128, "n must be < 128"
    mask = remove_all()

    for i in range(n):
        mask[order[i]] = 1

    return mask


if __name__ == '__main__':
    # mask = top(20)  # This line allows you to keep the top activated 20 elements
    # mask = change(20)  # This line allows you to keep the top changing 20 elements
    mask = tsne_1d_projection(127)  # This line allows you to keep the top tsne_1d_projection 50 elements
    # mask = remove_all()  #This removes all elements.

    data = gen_classic(mask, "offi_kitem.json")

def reduce(scenario, model):
    # mask = top(20)  # This line allows you to keep the top activated 20 elements
    # mask = change(20)  # This line allows you to keep the top changing 20 elements
    mask = tsne_1d_projection(127)  # This line allows you to keep the top tsne_1d_projection 50 elements
    # mask = remove_all()  #This removes all elements.

    data = gen_classic(mask, "result_"+model+".json", scenario=scenario, model=model)

