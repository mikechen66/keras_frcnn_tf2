import numpy as np
import pdb
import math
from . import data_generators
from keras import backend as K
import copy


# R —>result, C —> cfg = config.Config() 
def calc_iou(R, img_data, C, class_mapping): 
    """
    calc_iou找出剩余region对应gt里重合度最高的bbox从而获得model_classifier的数据和标签；X2保留所有的背景和匹配bbox
    的框；Y1是类one-hot转码；Y2是对应类标签及回归要学习的坐标位置;IouS用于debug；把(x1,y1,x2,y2)转换为(x,y,w,h)格式
    :param R: R=[boxes,probs], 通过rpn网络输出结果，选取出对应的rois,shape=(rois个数，4)
    :param img_data： image图片数据，经过相关预处理后的原始数据，格式如下
        {'width': 500,
         'height': 500,
         'bboxes': [{'y2': 500, 'y1': 27, 'x2': 183, 'x1': 20, 'class': 'person', 'difficult': False},
                    {'y2': 500, 'y1': 2, 'x2': 249, 'x1': 112, 'class': 'person', 'difficult': False},
                    {'y2': 490, 'y1': 233, 'x2': 376, 'x1': 246, 'class': 'person', 'difficult': False},
                    {'y2': 468, 'y1': 319, 'x2': 356, 'x1': 231, 'class': 'chair', 'difficult': False},
                    {'y2': 450, 'y1': 314, 'x2': 58, 'x1': 1, 'class': 'chair', 'difficult': True}], 
         'imageset': 'test',
        'filepath': './datasets/VOC2012/JPEGImages/000910.jpg'
        }
    :param C: 保存相关的配置文件config.Config()
    :param class_mapping: 每个类的标签，它存放样本类字符串和数字对应关系如'bg'：0
    :return: 
        np.expand_dims(X, axis=0), 
        np.expand_dims(Y1, axis=0), 
        np.expand_dims(Y2, axis=0), 
        IoUs
    """
    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # 调用数据生成器额get_new_img_size()函数获得缩放图像的维度
    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

    # 这里与calc_rpn()中基本一致，gta是指ground truth anchor, 参数shape=4维度
    gta = np.zeros((len(bboxes), 4)) 

    # for循环获得真实标注在特征图上的GT box坐标进行缩放，把最短边规整到指定的长度后，相应的边框长度也需要发生变化；请注
    # 意gta的存储形式是(x1,x2,y1,y2)而不是(x1,y1,x2,y2)
    for bbox_num, bbox in enumerate(bboxes): 
        # gta[bbox_num, 0] = (40*(600/800))/16 = int(round(1.875)) = 2 (x in feature map)
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / C.rpn_stride))

    # 对以下５个赋空值列表
    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] 

    # 遍历所有预选框R(roi)而无需做调整，因为RPN网络预测的框是基于最短框被调整后的，shape[0]表示宽(rows)
    for ix in range(R.shape[0]):
        # 请注意R=[boxes,probs]，ix包括x-axis上所有点, gta的保存格式为
        (x1, y1, x2, y2) = R[ix, :]　
        # int()方法求整，round()方法返回浮点数x1,y1,x2,y2的四舍五入值
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        # 存储当前roi(候选框)与所有真实标注框之间的最优iou值
        best_iou = 0.0
        # 当前roi(候选框)对应的最优候选框index,请注意索引值为-1表示放在最后
        best_bbox = -1

        # 把每一个预选框与所有的bboxes求交并比，记录最大交并比，用于确定该预选框的类别
        for bbox_num in range(len(bboxes)):
            # 调用数据生成器的iou()方法, gta是相对于原图缩小比例的bbox，而x1,x2,y1,y2是生成框
            curr_iou = data_generators.iou([gta[bbox_num,0], gta[bbox_num,2], gta[bbox_num,1], gta[bbox_num,3]],
                                           [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        # 如果对于某个框匹配当前bbox的重叠率小于0.3，那么这个框就被扔掉；self.classifier_min_overlap = 0.1
        if best_iou < C.classifier_min_overlap:
            continue
        # 当大于最小阈值时，则保留相关的边框信息
        else:
            w = x2 - x1
            h = y2 - y1
            # 把x1,y1,w,h列表添加到x_roi
            x_roi.append([x1, y1, w, h])
            # IoUs仅用于bebug
            IoUs.append(best_iou)

            # 当在最小和最大之间，则认为背景有必要进行训练; 
            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # 硬负例: 在对负样本分类时label与prediction之差loss较大的样本，即容易把负样本看成正样本的那些样本，
                # 例如roi中没有物体而全是背景
                cls_name = 'bg'
            # 当大于最大阈值时获得物体，则计算其边框回归梯度，cxg is cx gta? self.classifier_max_overlap = 0.5
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0 
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        # 获得该类对应的数字和标签0,1,2...
        class_num = class_mapping[cls_name]
        # 用0类填空标签
        class_label = len(class_mapping) * [0]
        # 把该数字对应的地方置为1即one-hot
        class_label[class_num] = 1
        # 把该类别加入到y_class_num，
        y_class_num.append(copy.deepcopy(class_label))
        # coords是用于存储边框回归梯度
        coords = [0] * 4 * (len(class_mapping) - 1)
        # labels决定是否要加入计算loss中
        labels = [0] * 4 * (len(class_mapping) - 1)

        # 若它不是背景，则计算相应的回归梯度
        if cls_name != 'bg':
            label_pos = 4 * class_num
            # s是scale(缩放)的缩写，config脚本中self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
            sx, sy, sw, sh = C.classifier_regr_std
            # 调整坐标相当于coords回归要学习的内容
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
            # deepcopy()改变保存位置
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # 把以上x_roi作为参数传入数组函数np.array()
    X = np.array(x_roi)
    # Y1是bboxes的独热编码--从以上的x_roi(x)
    Y1 = np.array(y_class_num)
    # Y2是相应类的label标签和要学习的坐标位置gt bboxes 
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    # X是缩放后的原始图像，最小边为600(用于share_layers;X2为IOU>0.7的roi; Y1类的顺序为(1,xxx,21)，类号为one_hot;
    # Y2:[np.array(y_class_regr_label)，np.array(y_class_regr_coordds)]包含一个独热的类标签和regr参数; IOUs
    # 只用于调试。
    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):
    """
    输入是一组原始信息(已被RPN网络的reg层输出调整过)，一组调整的参数(classifier模型输出的回归参数); 根据训练过程中回
    归参数的计算方式逆向计算; 通过classifier模型输出的回归参数再次调整。
    """
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    """
    传入的两个参数是A和regr, A(X)是根据anchor scale直接生成的nd_array。regr(T)是RPN网络reg层输出的张量，代表anchor 
    box需修正的偏移量, 通过RMP网络的回归层预测值调整anchor位置。在训练过程中，对于box中心点偏移量的定义是偏差÷高度/宽度；
    对于box边长偏移量的定义是(log(偏差))÷高度/宽度；在测试过程中反向操作如下函数：
    cx1 = tx * w + cx，w1 = np.exp(tw.astype(np.float64)) * w.
    :param X: 对于RexNet50， 维度为(4,38.50)
    :param T: 同上
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

　　　　 # np.round( )函数将小数化为整数。
        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)

        # 关于np.stack([x1, y1, w1, h1])，见官方文档: numpy.stack(), 默认沿axis=0方向堆叠，因此输出的维度还是
        # (4,38,50).
        return np.stack([x1, y1, w1, h1])

    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, overlap_thresh=0.9, max_boxes=300):
    """
    请谨慎处理: 改变boxes包含概率的方法以便无须在该方法中发送prob,当前的boxes实际上是[x1,y1,x2,y2,prob]格式
    非极大值抑制算法，提取出300个anchors作为输入roipooling层的roi，假如当前有10个anchor，根据正样本的概率值进行升序
    排序为[A,B,C,D,E,F,G,H,I,J]
    1.从具有最大概率的anchor J开始，计算其余anchors与J之间的iou值
    2.若iou值大于overlap_thresh阈值，则删除掉把当前J重新保留下来，若D、F与J之间的iou大于阈值，则直接舍弃，同时把J重新
    保留并原始数组中删除掉。
    3.在剩余的[A,B,C,E,G,H]中继续选取最大的概率值对应的anchors,然后重复上述过程。
    4.当数组为空或保留下来的anchor个数达到设定的max_boxes，则停止迭代，最终保留的anchor是需要的。
    :param boxes: 经过rpn网络后生成的所有候选框,shape=(anchor个数，4)
    :param probs: rpn网络分类层的输出值，value对应正例样本的概率，shape=(anchor个数，)
    :param overlap_thresh:  iou阈值
    :param max_boxes: 最大提取的roi个数
    :return boxes
    """
    if len(boxes) == 0:
        return []
    # 归一化到np.array
    boxes = np.array(boxes)
    # 获取bboxes的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    area = (x2 - x1) * (y2 - y1)
    # 所有anchor根据概率值进行升序排序，最后一个元素是概率？
    indexes = np.argsort([i[-1] for i in boxes])

    # 最后一个索引为当前idxs中具体最大概率值(是否为正例)的anchor的索引, 保留当前anchor对应索引
    while len(indexes) > 0:
        last = len(indexes) - 1
        # 最后一个索引为当前idxs中具有最大概率值(是否为正例)的anchor的索引
        i = indexes[last]
        # pickle保留当前anchor对应的索引
        pick.append(i)

        # 计算当前选取出的anchor与其它anchor之间的交集
        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])
        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        # 当前选取出来的索引对应的anchor与其它anchor之间的交集
        area_int = ww_int * hh_int
        # 发现交集
        area_union = area[i] + area[indexes[:last]] - area_int

        # 计算并交比(IOU)
        overlap = area_int / (area_union + 1e-6)

        # 在idxs索引列表中中删除掉与当前选取出来的anchor之间IOU大于overlap_thresh阈值的。
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        # 若当前保留的anchor个数达到max_boxes，则直接跳出迭代
        if len(pick) >= max_boxes:
            break

    # 仅返回使用整数数据类型选取的边界框
    boxes = boxes[pick]

    return boxes


def rpn_to_roi(rpn_layer, regr_layer, cfg, data_format, use_regr=True, max_boxes=300, overlap_thresh=0.9):
　　 """
    把rpn层转换为roi bboxes(用于VGG或ResNet50), Args：（num_anchors = 9）
    :param rpn_layer：rpn分类的输出层形状(1，feature_map.height，feature_map.width，num_anchors); 若调整大小的
        图像是400宽和300高，则可能是(1、18、25、18);
    :param regr_layer：rpn回归的输出层形状(1，feature_map.height，feature_map.width，num_anchors)； 若调整大小
        的图像是400宽和300高，则可能是(1、18、25、72);
    :param cfg：config.Config（）
    :param data_format: K.image_data_format()
    :param use_regr：可在rpn中使用bboxes回归
    :param max_boxes：NMS非最大抑制的最大bboxes数
    :param overlay_thresh：若NMS中的iou大于此阈值，请删除该框
    :return：
        result：来自非最大抑制的盒子(shape = (300，4)), box：bbox的坐标(在特征图上）
    """
    # self.config.std_scaling在config.py文件中默认设置为4.0
    regr_layer = regr_layer / cfg.std_scaling　
    # anchor_box_scales = [128, 256, 512]
    anchor_sizes = cfg.anchor_box_scales 
    # self.anchor_box_ratios = [[1, 1], [1，2], [2, 1]
    anchor_ratios = cfg.anchor_box_ratios 
    # 若一张图片的输入的shape=[1,h,w,512]的shape[0]=1，则该条件不满足就报错；请注意shape[0]=1是batch数量
    assert rpn_layer.shape[0] == 1

    # -if dim_ordering == 'th':
    if data_format == 'channels_first': 
        (rows, cols) = rpn_layer.shape[2:]
    # -elif dim_ordering == 'tf':
    elif data_format == 'channels_last':
        (rows, cols) = rpn_layer.shape[1:3]

    # 当前的anchor标记为0, 回归总共有9个通道(序号从0到8)
    curr_layer = 0 

    # 定义所有锚框的中心坐标和宽高的矩阵，因为回归梯度由4个维度，所以np.zeros()的第1个参数为维度4，每个维度来存放一个坐标， 
    # A[0,...]=x1, A[1,...]=y1, A[2,...]=x2, A[3,...]=y2, np.zeros()默认64位浮点？
    # -if dim_ordering == 'tf':
    if data_format == 'channels_last':
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
    # -elif dim_ordering == 'th':
    elif data_format == 'channels_first': 
        A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

    # 两个for循环分别遍历anchor_sizes和anchor_ratios,获得anchor的宽和高
    for anchor_size in anchor_sizes: 
        for anchor_ratio in anchor_ratios: 
            anchor_x = (anchor_size * anchor_ratio[0]) / cfg.rpn_stride 
            anchor_y = (anchor_size * anchor_ratio[1]) / cfg.rpn_stride 
            # -if dim_ordering == 'th':
            if data_format == 'channels_first':
                # 获取当前通道对应的回归值(4个一组)，共有9 anchors—>curr_layer: 0~8, 请注意regr_layer[]列表有4个
                # 数值，在扣除一个数值(num_anchors)后，regr仅有3个数值(18,25,4),其中4为通道数
                regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            else:
                # 在ResNet50中，当某个维度的值为一个时，那么新变量就减小一个维度，regr_layer维度为(18,18,25,4)，把
                # 当前缩放对应的4个坐标的回归信息提取到变量regr中，regr的维度为(18,25,4)，调整维度顺序(channels调到
                # 第一个维度)，最终regr的维度为(4,18,25)
                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
                # 通道从后面转换到前面(W,H,C)->(C,W,H)，本例为(0,1,2)—>(2,0,1)
                regr = np.transpose(regr, (2, 0, 1)) 
            # 把特征图转成网格矩阵，X是每一行的x坐标矩阵，即X里的每一行是x的坐标； Y是每一行的y坐标矩阵，每一行是y坐标，
            # 从0行0列开始
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows)) 

            # 填充锚框坐标矩阵，4个值为计算锚框修正方便，暂时为中心点以及宽和高，后面修正完会改成4个坐标值
            A[0, :, :, curr_layer] = X - anchor_x / 2  
            A[1, :, :, curr_layer] = Y - anchor_y / 2  
            A[2, :, :, curr_layer] = anchor_x          
            A[3, :, :, curr_layer] = anchor_y          

            # 进行锚框的回归修正，即把RPN的回归系数用于锚框上，回归后变成预测框，希望预测框和真实框的差距越小越好，以便
            # 训练回归系数后的A称为预测框(或称为锚框的回归版)
            if use_regr:
                # 对anchor进行变换
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr) 
            # 调整变换anchor的位置，anchor的宽度和高度都不能小于1，
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            # 宽和高分别加上坐标，等于右下角的坐标——即把宽和高的位置换成右下角坐标x2,y2，现在A里面的值是x1,y1,x2,y2
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]
            # 修剪anchor，anchor的x1,y1不能超出feature map范围，xi,y1不能小于0
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            # 修剪anchor, anchor的x2,y2不能超过宽和高的边界，最大就是边界值-1
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

            # 当前的anchor标记标记增１
            curr_layer += 1 

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # 找到那些box坐标无效(invalid)的ids
    ids = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    # 删掉这些box, all_boxes(xxx,1)成为一个行向量
    all_boxes = np.delete(all_boxes, ids, 0)
    # 删掉这些box对应的probs概率
    all_probs = np.delete(all_probs, ids, 0)

    # Guess boxes and prob are all 2d array and will concat them
    all_boxes = np.hstack((all_boxes, np.array([[p] for p in all_probs])))
    result = non_max_suppression_fast(all_boxes, overlap_thresh=overlap_thresh, max_boxes=max_boxes)
    # 这个列表很奇怪，Omit the last column which is prob
    result = result[:, 0: -1]

    # 返回有效box中具有最大概率包含物体的预box(删除IOU并交比较高的box(xxx,4))，其形式是(x1,y1,x2,y2）
    return result