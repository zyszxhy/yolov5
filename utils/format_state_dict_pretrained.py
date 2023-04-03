
from collections import OrderedDict

# weights = 'yolov5/weights_in_coco/yolov5l.pt'
# ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
# csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32

def format_state_dict(csd_rgb, csd_ir):

    my_yolo_state_dict = OrderedDict()

    for key, value in csd_rgb.items():

        layer_id = int(key.split('.')[1])

        if layer_id>=0 and layer_id<=4:
            my_yolo_state_dict[key] = value
            layer_id_ir = str(layer_id + 5)
            key_ir = key.replace(str(layer_id), layer_id_ir, 1)
            # my_yolo_state_dict[key_ir] = value
            my_yolo_state_dict[key_ir] = csd_ir[key]

        elif layer_id>=5 and layer_id<=6:
            layer_id_rgb = str(layer_id + 5)   # 17 15 11
            key_rgb = key.replace(str(layer_id), layer_id_rgb, 1)
            my_yolo_state_dict[key_rgb] = value
            layer_id_ir = str(layer_id + 7)    # 19 17 13
            key_ir = key.replace(str(layer_id), layer_id_ir, 1)
            # my_yolo_state_dict[key_ir] = value
            my_yolo_state_dict[key_ir] = csd_ir[key]
        
        elif layer_id>=7 and layer_id<=9:
            layer_id_rgb = str(layer_id + 7)   # 31 27 19
            key_rgb = key.replace(str(layer_id), layer_id_rgb, 1)
            my_yolo_state_dict[key_rgb] = value
            layer_id_ir = str(layer_id + 10)    # 34 30 22
            key_ir = key.replace(str(layer_id), layer_id_ir, 1)
            # my_yolo_state_dict[key_ir] = value
            my_yolo_state_dict[key_ir] = csd_ir[key]

        elif layer_id>=10:
            layer_id_rgb = str(layer_id + 13)   # 55 49 37
            key_rgb = key.replace(str(layer_id), layer_id_rgb, 1)
            my_yolo_state_dict[key_rgb] = value
            # layer_id_ir = str(layer_id + 19)
            # key_ir = key.replace(str(layer_id), layer_id_ir, 1)
            # my_yolo_state_dict[key_ir] = value
    
    return my_yolo_state_dict