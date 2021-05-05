"""
# 计算y_rpn_overlap、y_is_box_valid和y_rpn_regr
  y_rpn_overlap和y_is_box_valid构成anchor的分类结果y_rpn_cls
  y_rpn_overlap和y_rpn_regr构成anchor的最优回归参数y_rpn_regr

# 输出：
    输出的结果包含所有anchor的结果，其中valid为1的是neg和pos类、0是中性，overlap为1的是有交叉的、0是无交叉的，因此有交叉
    的、valid为1的就是pos的anchor，在对应的位置记录了对应最优的回归参数， y_rpn_cls：所有anchor的valid和overlap标记；
    y_rpn_regr：所有anchor的overlap标记和部分最优的回归参数

# 核心方法def calc_rpn():
    def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    方法的作用是在原始图片上找到合法的Anchor，每一个feature map上的点都对应9个anchor，由于网络为已知，feature map和原图
    的对应关系为已知，例如feature map上的一个点对应原图上的16个像素，即使没有得到feature map也能在原图上根据feature map
    的大小设定anchor的数量，该函数去掉那些超出边界的Anchor，去掉那些和原始标注框(GTbox)无交集的Anchor，找到和GTbox重合度
    最好的anchor并得到bbox regression参数。

# for循环嵌套(5层)
    for anchor_size_idx in range(len(anchor_sizes)):
    for循环有5层，第１层是anchor大小，第２层是anchor比例，第３层是横向遍历特征图上(x)每个点，第４层是纵向遍历特征图上(y)每
    个点，第５层是遍历每个GTbox，这５层循环保证所有的anchor和所有的GTbox都比较一遍从而找到最优IOU交并比
"""


from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def iou(a, b):
    # a和b的重叠度iou应该是(x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width　# f为浮点值
        resized_height = int(f * height)
        resized_width = img_min_side     # 最小边的宽为600
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side
    return resized_width, resized_height


class SampleSelector:
    def __init__(self, class_count):
        # 忽略有零样本的类(Ignore the classes that have zero samples)
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        # Python的itertools.cycle()循环迭代器
        self.class_cycle = itertools.cycle(self.classes) 
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):
        class_in_img = False
        for bbox in img_data['bboxes']:
            cls_name = bbox['class']
            if cls_name == self.curr_class:
                class_in_img = True
                # 调用itertools.cycle()循环迭代器，参见以上第78行
                self.curr_class = next(self.class_cycle) 
                break
        if class_in_img:
            return False
        else:
            return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    """
    RPN生成regional proposal
    :param C: config.Config()配置文件
    :param width: img_data中的width
    :param height: img_data中的height
    :param img_data: 图片配置文件，包含信息如下：
        [0] = {'width': 500, 
               'height': 500,　
               'bboxes': [{'y2': 500, 'y1': 27, 'x2': 183, 'x1': 20, 'class': 'person', 'difficult': False}, 
                          {'y2': 500, 'y1': 2, 'x2': 249, 'x1': 112, 'class': 'person', 'difficult': False},
                          {'y2': 490, 'y1': 233, 'x2': 376, 'x1': 246, 'class': 'person', 'difficult': False},
                          {'y2': 468, 'y1': 319, 'x2': 356, 'x1': 231, 'class': 'chair', 'difficult': False},
                          {'y2': 450, 'y1': 314, 'x2': 58, 'x1': 1, 'class': 'chair', 'difficult': True}], 
               'imageset': 'test',
               'filepath': './datasets/VOC2012/JPEGImages/000910.jpg'}
    :param resized_width: 原始图片按要求缩放后的宽，例如以短边600像素等比缩放
    :param resized_height： 原始图片按要求缩放后的高，
    :param img_length_calc_function: 原始图片和特征图大小的对应关系，例如vgg由于采用4个pooling层且卷积层padding为1，
        不缩放图片，因此vgg得到的特征图的大小是原图的16分之一
    :return np.copy(y_rpn_cls), np.copy(y_rpn_regr)
    """
    # shared_layers共享卷积层把resize后的图片压缩16倍(默认)，经base_net处理的特征图尺度为(600/16,600/16)=(37,37)
    downscale = float(C.rpn_stride)
    # 如self.anchor_box_scales=[128,256,512]针对原图，原图在处理后为800*600，则512*512的anchor能覆盖大部分区域
    anchor_sizes = C.anchor_box_scales
    # self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]], C应为config.Config()
    anchor_ratios = C.anchor_box_ratios
    # 3种尺寸、3中缩放比例，一共是3x3=9种anchor
    num_anchors = len(anchor_sizes) * len(anchor_ratios)    

    # 计算图片经过共享卷积后输出的特征图的大小
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios) # anchor比例的数量
    
    # 初始化空的输出目标: 保存和gt是否重叠(正例还是反例)，根据IOU值、output_height、output_width对应在其中一种模板框下框的
    # 坐标，总体表示为在这个坐标下是否和gt重叠，重叠为1是正例，不重叠为0是反例，num_anchors表示数量，表示9重不同规模的框
    # 每个特征图中有9个anchor，每个anchor和GT是否交叠
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))　
    # 保存框是否有效
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors)) 
    # 保存框的4个坐标值
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4)) 

    # 图片上的bbox的数量
    num_bboxes = len(img_data['bboxes']) 
    # 每个bbox对应的候选anchor的数量
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    # 4为[jy, ix, anchor_ratio_idx, anchor_size_idx]
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int) 
    # float.32(第99、101、319行), TypeError: Input 'y' od 'Sub' op has type float32 that does not match  type 
    # int64 of argument 'x'
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32) 
    # 最优anchor的坐标, 4表示[x1_anc, x2_anc, y1_anc, y2_anc]
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)  
    # 最优anchor的bbox regression变换参数d(x)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32) 

    # 获得GT box coordinates并表示图像缩放，转换标签bbox调整大小的图像，只需根据resize/original_size改变比例; 请注意gta
    # (gt anchor)的存储形式是(x1,x2,y1,y2)而不是(x1,y1,x2,y2)
    gta = np.zeros((num_bboxes, 4))

    # 把最短边规整到指定的长度后，相应的边框长度需要发生变化
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # 获得GT box坐标并缩放表示image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
    
    # 设置rpn gt并迭代anchor size和raio从而获得所有可能的RPN: 对9种框进行遍历，里面包括对图片中所有物体的遍历，所有for循环
    # 次数len(anchor_sizes)遍历顺序，先遍历框，然后获得特征图和原始图框坐标，再与原始图中物体进行IOU计算，获得IOU最高的框
    for anchor_size_idx in range(len(anchor_sizes)):　
        for anchor_ratio_idx in range(n_anchratios):  
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]   
            
            # 第3个for横向遍历feature map上的点. 在x轴方向ix=[0,1,2,...,36]
            for ix in range(output_width):                  
                # 当前anchor box的x-coordinates， 
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2  
                
                # 删除跨越图像边界的框，仅当ix=4切jy=4时才形成一个完整的anchor，中心点坐标(72,72),anchor(8,8,136,136)
                if x1_anc < 0 or x2_anc > resized_width:
                    continue
                    
                # 第4层纵向遍历feature map上的每个点，在y轴方向jy=[0,1,2,...,36]
                for jy in range(output_height):　
                    # 当前anchor box的y-coordinates，downscale为特征图坐标映射到原图的比例
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # 删除跨越图像边界的框
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type表示锚是否应该是目标，这里为默认为'neg' 
                    bbox_type = 'neg'

                    # 这是(x,y)坐标最好的IOU和当前锚，请注意这不同于GT bbox的最佳IOU
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes): # 第5层依次遍历每个GT预选标注框
                        # 获得当前预选框GT和当前anchor box的IOU：若IOU越高，则候选框anchor和预选框gt的重叠比列越高且越接近gt
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], 
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # 若确实需要，则计算回归目标
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            # GTbox中心点的x坐标
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0  
                            # GTbox中心点的y坐标
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            # 当前anchor中心点的x坐标
                            cxa = (x1_anc + x2_anc)/2.0 
                            # 当前anchor中心点的y坐标
                            cya = (y1_anc + y2_anc)/2.0 

                            # 计算用于box regression的训练集的回归参数，用于输入到RPN中训练网络，这样在预测时就能根据训练结果
                            # 生成一个尽可能好的回归参数，tx,ty分别为两个框中心的宽高的距离与预gt选框的宽的比
                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc) 
                            # bbox的宽与预选框宽的比
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc)) 
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc)) 
                        
                        # 如果相交的不是背景，那么进行一系列更新
                        if img_data['bboxes'][bbox_num]['class'] != 'bg':
                            # 所有GT box应映射到anchor box，这样能跟踪哪个anchor box最好, gt预选框更新：交并比大于阈值是pos
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                 # 当前处理的feature map的点的坐标和最优anchor比例和大小
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                # 把当前curr_iou赋值给best_iou_for_box[]
                                best_iou_for_bbox[bbox_num] = curr_iou　
                                # 最优anchor的坐标
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]　
                                # 最优anchor的bbox regression变换参数d(x)
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th] 

                            # 若IOU>0.7，则把anchor设为正例(pos)
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                #只有bbx_type为'pos'，预选框gt数量才增１
                                num_anchors_for_bbox[bbox_num] += 1 
                                # 若该IOU是当前(x,y)和锚点位置的最佳IOU则更新回归层目标; best_iou_for_loc记录最大交并比的
                                # 值和对应的回归梯度,  临时保存best_iou和best_regr参数
                                if curr_iou > best_iou_for_loc: 
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # 若IOU是>0.3和<0.7，则它是模糊(neutral)而未包括在目标内因而被丢弃
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # 在neg和pos(负例和正例)之间是灰色区域
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # 打开或关闭输出取决于IOUs; box有neg,pos和neutral; neg:box的背景，pos:box有RPN,neutral:带RPN的普通box;
                    # y_is_box_valid：该预选框gt是否可用(nertual不可用), y_rpn_overlap：预选框是否有物体,y_rpn_regr:回归梯度；
                    # 当iou < 0.3时标记为neg，jy,ix表示在feature map上的坐标
                    if bbox_type == 'neg': 
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1 
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

    # 确保每个bbox至少有一个正例的RPN区域，乘法运算符*是什么意思
    for idx in range(num_anchors_for_bbox.shape[0]):
        # 每个bbox对应的候选anchor的数量， 当前标注框的候选anchor为0
        if num_anchors_for_bbox[idx] == 0: 
            # 值为-1说明不是best_anchor
            if best_anchor_for_bbox[idx, 0] == -1:　
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1 
            y_rpn_overlap[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

    # 把深度2的顺序改到第一位，给向量增加一个维度； 对于VGG—(37,37,9)->(9*37*37)
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1)) 
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    # 把pos与neg总是超过num_regions个的neg预选框置为不可用，请注意np.logical_and()的用法
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    # 有多少pos的anchor 
    num_pos = len(pos_locs[0]) 

    # 问题是RPN的neg区域比pos区域多，因此关闭一些neg区域并限制为256个区域，正例不能超过128
    num_regions = 256

    # 若pos数量大于num_regions/2=256，则随机选取128个anchor，小于128则全部保留
    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)　
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2

    # 若neg和pos的anchor总和大于256，则仅选取256个anchor
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    # 请注意np.copy()是深度拷贝，返回值是y_rpn_cls为否包含类，返回值y_rpn_regr为相应回归梯度，这两个值作为参数传递给get_anchor_gt()
    # 函数中的calc_rpn()
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
    """取iterator/generator并通过它们给出的next()方法使之成为线程安全(thread-safe)"""
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)        

    
def threadsafe_generator(f):
    """装饰器取一个生成器函数并使其为thread-safe."""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
    # 代码行all_img_data = sorted()在Python 3.5中无用，它用于Python 2.7
    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:
                # 调用sample_selector类中的skip函数
                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # 读取图像并选择是否添加增强, augument()调用augument脚本中的同名函数，其中参数C为(未明说的)配置参数
                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # 调用get_new_img_size()函数获得缩放的图像维数
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                # 缩放图像以便最小边的长度为600px(length = 600px)
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                except:
                    continue

                # 以平均像素为零中心(zero-center)对图像进行预处理
                x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)

                # y_rpn_regr.shape[1]为72, 72//2=36, 因此这里选后36个[36:]，为best reg，都乘一个标准差系数C.std_scaling=4.0从而扩大数据波动范围
                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

                # -if backend == 'tf':
                if backend == 'channels_last':
                    x_img = np.transpose(x_img, (0, 2, 3, 1)) # 若为tensorflow，x_img又变回(1,W,H,C)
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                # 每次遇到yield时函数会暂停、保存当前所有的运行信息并返回yield的值, 并在下一次执行next()方法时从当前位置继续运行，next()
                # 函数中X得到经过最小边600的缩放变换的原始图像，Y为所有框位置的类别(正例还是反例)，img_data为增强图像后的图像信息，请注意
                # Y的值[np.copy(y_rpn_cls), np.copy(y_rpn_regr)]中np.copy()为深拷贝, 拷贝前后的地址不一样。
                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(e)
                continue