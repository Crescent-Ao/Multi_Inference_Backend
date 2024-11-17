from yolov6.distiller.distill_cfg import CFG as cfg
from yolov6.distiller.DKD import *
from yolov6.distiller.FitNet import *
from yolov6.distiller.AT import *
from yolov6.distiller.review_kd import *
from yolov6.distiller.NST import *
from yolov6.distiller.SP import *
def build_distiller(distill_method:str,cfg = cfg):
    """
    Args:
        distill_method (str):蒸馏的范式
    """
    method = ["DKD","FitNet","ReviewKD","NST","AT","SP"]
    if distill_method not in method:
        print("The current distillation method has not been implemented")
        raise NotImplementedError
    if distill_method == "DKD":
        distiller = DKD(cfg)
        distiller.name = "DKD"
    elif distill_method == "FitNet":
        distiller = FitNet(cfg)
        distiller.name = "FitNet"
    elif distill_method == "AT":
        distiller = AT(cfg)
        distiller.name = "AT"
    elif distill_method == "ReviewKD":
        distiller = ReviewKD(cfg)
        distiller.name = "ReviewKD"
    elif distill_method == "NST":
        distiller = NST(cfg)
        distiller.name = "NST"
    elif distill_method == "SP":
        distiller = RKD(cfg)
        distiller.name = "SP"
   
    return distiller