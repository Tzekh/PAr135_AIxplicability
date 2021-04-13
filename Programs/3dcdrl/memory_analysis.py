import reduce
import json
from PIL import Image

def gen_data():
    scenario = "corridor.cfg"


    reduce.reduce(scenario,"base_line")
    reduce.reduce(scenario,"model_final")

def mem_dist(start=[0, 0], end = 525):

    mem1 = []
    mem2 = []
    dist = []

    with open('result_base_line.json') as base_json:
        data1 = json.load(base_json)
        for vect in data1['hiddens']:
            mem1.append(vect)

    with open('result_model_final.json') as final_json:
        data2 = json.load(final_json)
        for vect in data2['hiddens']:
            mem2.append(vect)

    for i in range(end):
        dist.append([])
        for j in range(128):
            dist[i].append(abs(float(mem1[i+start[0]][j])-float(mem2[i+start[1]][j])))
    im = 0
    im = Image.new('P', (end, 128))
    for i in range(128):
        for j in range(end):
            value = float(mem2[j][i])
            if value >= 0:
                im.putpixel((j, i), (255-int(value*255), 255, 255, 255))
    im.save("comparaison.png")
    im2 = Image.open("comparaison.png")
    for i in range(128):
        for j in range(end):
            value = float(mem2[j][i])
            if value < 0:
                im.putpixel((j, i), (255, 255-int(abs(value)*255),  255, 255))
    im2.save("comparaison.png")

    return dist
gen_data()
mem_dist(end=80)