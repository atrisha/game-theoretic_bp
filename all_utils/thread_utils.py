'''
Created on May 21, 2020

@author: Atrisha
'''
from itertools import islice
import concurrent.futures
import threading
import multiprocessing as mp
import numpy as np
import logging
import constants
logging.basicConfig(format='%(levelname)-8s %(funcName)s-2s  %(module)s: %(message)s',level=logging.INFO)
log = constants.common_logger

class DictProcessor:
    
    def __init__(self,func,params,all_chunks,dict_to_process):
        self.func_to_exec = func
        self.func_to_exec_params = params
        self.all_chunks = [x for x in all_chunks]
        self.dict_to_process = dict_to_process
    
    @classmethod
    def fromFlattenedDict(cls,dict_to_process,func_name,func_params,max_size):
        def chunks(dict_to_process,max_size):
            data = dict_to_process
            SIZE = max_size
            it = iter(data)
            for i in range(0, len(data), SIZE):
                yield {k:data[k] for k in islice(it, SIZE)}
        dict_to_process = dict_to_process
        max_size = max_size
        dict_chunks = chunks(dict_to_process,max_size)
        func_to_exec = func_name
        return cls(func_to_exec,func_params,dict_chunks,dict_to_process)
        
    def execute_threads(self):
        all_chunks = self.all_chunks
        processed_dict = dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_chunks)) as executor:
            future_to_dict = {executor.submit(self.func_to_exec, chunk, self.func_to_exec_params): chunk for chunk in all_chunks}
        for future in concurrent.futures.as_completed(future_to_dict):
            res_dict = future.result()
            for k,v in res_dict.items():
                processed_dict[k] = v
        return processed_dict
    
    def execute_threads_callback(self,callback,callback_args):
        all_chunks = self.all_chunks
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_chunks)) as executor:
            future_to_dict = {executor.submit(self.func_to_exec, chunk, self.func_to_exec_params): chunk for chunk in all_chunks}
        for future in concurrent.futures.as_completed(future_to_dict):
            res_obj = future.result()
            callback_res = callback(res_obj,callback_args)
        return callback_res
                
    def execute_mp(self):
        all_chunks = self.all_chunks
        num_core = 8
        pool = mp.Pool(num_core)
        manager = mp.Manager()
        processed_dict = manager.dict()
        chunk_len = len(all_chunks)
        batches = np.arange(0,chunk_len,num_core)
        for b_i in batches:
            jobs = []
            c_is = np.arange(b_i,min(b_i+num_core,chunk_len))
            for pc_ix,c_i in enumerate(c_is):
                p = mp.Process(target=self.func_to_exec, args=(all_chunks[c_i],processed_dict))
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
        processed_dict = {k:v for k,v in processed_dict.items()}
        f=1
        return processed_dict
    
    def execute_mp_callback(self,callback,callback_args):
        all_chunks = self.all_chunks
        num_core = 12
        pool = mp.Pool(num_core)
        manager = mp.Manager()
        processed_dict = manager.dict()
        chunk_len = len(all_chunks)
        batches = np.arange(0,chunk_len,num_core)
        ct = 0
        for b_i in batches:
            ct += 1
            log.info('processed batch '+str(ct)+'/'+str(len(batches)))
            jobs = []
            c_is = np.arange(b_i,min(b_i+num_core,chunk_len))
            for pc_ix,c_i in enumerate(c_is):
                p = mp.Process(target=self.func_to_exec, args=(all_chunks[c_i],self.func_to_exec_params,processed_dict))
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
        callback_res = callback(processed_dict,callback_args)
        return callback_res
        
    def execute(self):
        all_chunks = self.all_chunks
        all_futures = []
        
        if len(self.dict_to_process) < 300:
            '''use multithreading '''
            processed_dict = self.execute_threads()
        else:
            '''use multiprocessing '''
            processed_dict = self.execute_mp()
        return processed_dict
        
class CustomMPS():
    
    def execute_no_return(self,func,list_params):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_dict = {executor.submit(func, chunk): chunk for chunk in list_params}
            for future in concurrent.futures.as_completed(future_to_dict):
                future.result()
            return True
                
    def execute_with_callback(self,func,list_params,callback):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_dict = {executor.submit(func, chunk): chunk for chunk in list_params}
            for future in concurrent.futures.as_completed(future_to_dict):
                res = future.result()
                callback(res[0],res[1])
                

class CustomThreaExs():
    
            
    def execute_no_return(self,func,list_params):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(list_params)) as executor:
            future_to_dict = {executor.submit(func, chunk): chunk for chunk in list_params}
            for future in concurrent.futures.as_completed(future_to_dict):
                future.result()
            return True
                
    def execute_with_callback(self,func,list_params,callback):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(list_params)) as executor:
            future_to_dict = {executor.submit(func, chunk): chunk for chunk in list_params}
            for future in concurrent.futures.as_completed(future_to_dict):
                res = future.result()
                callback(res[0],res[1])
                
class ThreadSafeObject:
    """
    A class that makes any object thread safe.
    """
    
    def __init__(self, obj):
        """
        Initialize the class with the object to make thread safe.
        """
        self.lock = threading.RLock()
        self.object = obj
        
    def __getattr__(self, attr):
        self.lock.acquire()
        def _proxy(*args, **kargs):
            self.lock.acquire()
            answer = getattr(self.object, attr)(*args, **kargs)
            self.lock.release()
            return answer
        return _proxy
        