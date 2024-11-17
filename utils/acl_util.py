import os  
import acl
import acllite_utils as utils
import constants as const
from acllite_model import AclLiteModel
from acllite_resource import *
from acllite_resource import _ResourceList
import torch

class AclLiteResource:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None
        
    def init(self):
        print("init resource stage:")
        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        
        utils.check_ret("acl.rt.set_device", ret)
        
        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)
        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)
        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)
    
        print("Init resource success")
    def clean(self):
        try:
            resource_list.destroy()
            print("acl resource release all resource")
            if self.stream:
                print("acl resource release stream")
                acl.rt.destroy_stream(self.stream)
            if self.context:
                print("acl resource release context")
                acl.rt.destroy_context(self.context)
            print("Reset acl device ", self.device_id)
            acl.rt.reset_device(self.device_id)
            print("Release acl resource success")
            acl.finalize()
        except Exception as e:
            print(f"Error during resource cleanup: {e}")
        
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass  # 忽略在 __del__ 中可能发生的任何错误

