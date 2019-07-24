"""

@file  : label_map_util.py

@author: xiaolu

@time  : 2019-07-24

"""
import json


def create_category_index(categories):
    '''
    我们需要进一步整理　整理成字典
    :param categories: 第二个函数整理的结果　即：[{'id': item['id'], 'name': name}, {'id': item['id'], 'name': name}, []...]
    :return: {id1: name1, id2: name2, id3: name3.....}
    '''
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat   #
    return category_index


def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    '''
    整理成id和真实标签的字典集合的列表
    :param label_map: 预处理过后的标签数据
    :param max_num_classes: 最大类别数
    :param use_display_name: 是否展示真实的名字
    :return: [{'id': item['id'], 'name': name}, {'id': item['id'], 'name': name}, []...]
    '''
    categories = []
    for item in label_map:
        if use_display_name and 'display_name' in item:
            name = item['display_name']   # 真实的名字
        else:
            name = item['name']   # 如果没有真实的名字　可以用name属性值代替
        categories.append({'id': item['id'], 'name': name})
    return categories


def load_labelmap(path):
    '''
    给定标签存储的路径　整理标签数据
    :param path: 路径
    :return: [{'name': '/m/01g317', 'id': 1, 'display_name': 'person'}, {} ...]
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
        result = []
        cache = ''
        for line in lines:
            line = line.strip().strip('\n')
            if line.find('item') == 0:
                cache += '{'
            elif line.find('}') == 0:
                cache = cache[:-1] + '}'
                result.append(json.loads(cache))
                cache = ''
            else:
                line = line.split(':')
                line[0] = '"' + line[0] + '"'
                line = ':'.join(line)
                cache += line + ','
    return result   # [{'name': '/m/01g317', 'id': 1, 'display_name': 'person'}, {} ...]

#
# if __name__ == '__main__':
#     label_path = '../ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt'
#     result = load_labelmap(label_path)
#     print(result)


