import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from centernet import CenterNet
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

if __name__ == "__main__":
    '''
    Ao contrário do AP (Average Precision), Recall e Precision são conceitos de área. Portanto, quando o valor de limiar (Confidence) 
    é diferente, os valores de Recall e Precision da rede são diferentes.
    Por padrão, o Recall e Precision calculados por este código representam os valores correspondentes de Recall e Precision quando o 
    limiar (Confidence) é 0,5.

    Limitado pelo princípio de cálculo do mAP, a rede precisa obter quase todas as caixas de previsão ao calcular o mAP, para que os 
    valores de Recall e Precision em diferentes condições de limiar possam ser calculados.
    Portanto, o número de caixas txt em map_out/detection-results/ obtidas por este código é geralmente maior do que o da previsão direta. 
    O objetivo é listar todas as possíveis caixas de previsão.
    '''
    #------------------------------------------------- -------------------------------------------------- ---------------#
    # map_mode é usado para especificar o que é calculado quando o arquivo é executado
    # map_mode é 0 para representar todo o processo de cálculo do mAP, incluindo obtenção dos resultados de previsão, obtenção das caixas reais e cálculo do VOC_map.
    # map_mode é 1 para apenas obter o resultado de previsão.
    # map_mode é 2 para apenas obter a caixa real.
    # map_mode é 3 significa apenas calcular VOC_map.
    # map_mode é 4 para usar a biblioteca COCO para calcular o mAP de 0,50:0,95 do conjunto de dados atual. É necessário obter o resultado de previsão, obter a caixa real e instalar o pycocotools
    #------------------------------------------------- -------------------------------------------------- ----------------#
    map_mode = 0
    #------------------------------------------------- --------------------------------------#
    # O classes_path aqui é usado para especificar a categoria que precisa medir o VOC_map
    # Geralmente, é consistente com o classes_path usado para treinamento e previsão
    #------------------------------------------------- --------------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    #------------------------------------------------- --------------------------------------#
    # MINOVERLAP é usado para especificar o mAP0.x que você deseja obter, qual é o significado de mAP0.x, por favor, pesquise.
    # Por exemplo, para calcular o mAP0.75, você pode definir MINOVERLAP = 0.75.
    #
    # Quando a sobreposição entre uma caixa prevista e a caixa real é maior que MINOVERLAP, a caixa prevista é considerada uma amostra positiva, caso contrário, é uma amostra negativa.
    # Portanto, quanto maior o valor de MINOVERLAP, mais precisa a caixa de previsão deve ser prevista para ser considerada uma amostra positiva. Nesse momento, o valor de mAP calculado é menor.
    #------------------------------------------------- --------------------------------------#
    MINOVERLAP = 0.5
    #------------------------------------------------- --------------------------------------#
    # Limitado pelo princípio de cálculo do mAP, a rede precisa obter quase todas as caixas de previsão ao calcular o mAP, para que o mAP possa ser calculado
    # Portanto, o valor de confiança deve ser definido o mais baixo possível para obter todas as caixas de previsão possíveis.
    #
    # Esse valor geralmente não é ajustado. Como o cálculo do mAP precisa obter quase todas as caixas de previsão, a confiança aqui não pode ser alterada casualmente.
    # Para obter os valores de Recall e Precision sob diferentes limiares, modifique o score_threhold abaixo.
    #------------------------------------------------- --------------------------------------#
    confidence = 0.02
    #------------------------------------------------- --------------------------------------#
    # O tamanho do valor de supressão não máxima usado na previsão, quanto maior a supressão não máxima, menos rigorosa é a supressão não máxima.
    #
    # Esse valor geralmente não é ajustado.
    #------------------------------------------------- --------------------------------------#
    nms_iou = 0.5
    #------------------------------------------------- -------------------------------------------------- ------------#
    # Recall e Precision não são um conceito de área como AP, portanto, quando o valor de limiar é diferente, os valores de Recall e Precision da rede são diferentes.
    #
    # Por padrão, o Recall e Precision calculados por este código representam os valores correspondentes de Recall e Precision quando o valor de limiar é 0,5 (aqui definido como score_threhold).
    # Como o cálculo do mAP precisa obter quase todas as caixas de previsão, a confiança definida acima não pode ser alterada casualmente.
    # Um score_threhold é definido especificamente aqui para representar o valor de limiar e, em seguida, são encontrados os valores de Recall e Precision correspondentes ao valor de limiar ao calcular o mAP.
    #------------------------------------------------- -------------------------------------------------- ------------#
    score_threhold = 0.5
    #------------------------------------------------- ------#
    # map_vis é usado para especificar se a visualização do cálculo do VOC_map está habilitada
    #------------------------------------------------- ------#
    map_vis = False
    #------------------------------------------------- ------#
    # Aponta para a pasta onde o conjunto de dados VOC está localizado
    # Por padrão, aponta para o conjunto de dados VOC no diretório raiz
    #------------------------------------------------- ------#
    VOCdevkit_path = 'VOCdevkit'
    #------------------------------------------------- ------#
    # A pasta de saída do resultado, o padrão é map_out
    #------------------------------------------------- ------#
    map_out_path = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Carregar modelo.")
        centernet = CenterNet(confidence = confidence, nms_iou = nms_iou)
        print("Modelo carregado.")

        print("Obter resultado de previsão.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            centernet.get_map_txt(image_id, image, class_names, map_out_path)
        print("Resultado de previsão obtido.")
    if map_mode == 0 or map_mode == 2:
        print("Obter resultado de referência.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj. find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Resultado de referência obtido.")

    if map_mode == 0 or map_mode == 3:
        print("Obter VOC_map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("VOC_map obtido.")

    if map_mode == 4:
        print("Obter VOC_map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("VOC_map obtido.")
