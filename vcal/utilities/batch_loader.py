import torch
from itertools import zip_longest, cycle



class MultiSpaceBatchLoader(object):
    def __init__(self,*args,out_device=None):
        self.loaders = list(args)
        self.out_device = out_device
        self.instrumental_iterable = None

        self.n_over_m = [n/m for n,m in zip(self.dataset_sizes,self.batch_sizes)]
        self.one_over_n = [1/n for n in self.dataset_sizes]
        self.nspaces = len(self.loaders)

    # Method out_dims return the number of columns (or size of last dimension) of the last tensor of each "sub" loaders
    # contained in the MultiSpaceBatchLoader object.
    # The last tensor is generally the output, thus the name.
    @property
    def out_dims(self):
        res = []
        for l in self.loaders:
            try:
                res += [l.dataset.tensors[-1].size(-1)]
            except AttributeError:
                res += [l.dataset.dataset.tensors[-1].size(-1)] # in case l.dataset is a subset
        return res

    def setup_iterable(self):
        if self.cycle:
            iter_load_list = [cycle(iter(l)) for l in  self.loaders[:-1]]+[iter(self.loaders[-1])]
            it = zip(*tuple(iter_load_list))
        else:
            it = zip_longest(*(iter(l) for l in self.loaders))
        self.instrumental_iterable = it

    def __iter__(self):
        self.setup_iterable()
        return self

    def iterable(self,cycle=True,out_device=None):
        self.cycle = cycle
        self.out_device = out_device
        return self.__iter__()

    def __next__(self):
        data = next(self.instrumental_iterable)
        inputs  = list()
        outputs = list()
        for space in data:
            if space is None:
                inputs  += [None]
                outputs  += [None]
            else:
                inputs  += [x.to(self.out_device) for x in space[:-1]]
                outputs += [space[-1].to(self.out_device)]
        res = tuple(inputs),tuple(outputs)
        return res 
    
    def __len__(self):
        return len(self.loaders)
    @property
    def batch_sizes(self):
        return (l.batch_size for l in self.loaders)
    @property
    def datasets(self):
        return (l.dataset for l in self.loaders)
    @property
    def dataset_sizes(self):
        return (len(l.dataset) for l in self.loaders)

class SingleSpaceBatchLoader(MultiSpaceBatchLoader):
    def __init__(self,loader,out_device=None,cat_inputs=False):
        super(SingleSpaceBatchLoader,self).__init__(loader,out_device=out_device)
        l = self.loaders[0]
        self.dataset_size = len(l.dataset)
        self.batch_size   = min(self.dataset_size,l.batch_size)
        self.n_over_m = self.dataset_size/self.batch_size
        self.one_over_n = 1/self.dataset_size
        self.cat_inputs = cat_inputs

    def __next__(self):
        data, = next(self.instrumental_iterable)       
        if self.cat_inputs:
            inputs = torch.cat(tuple(d for d in data[:-1]),-1)
            return (inputs.to(self.out_device),data[-1].to(self.out_device))
        else:
            return (d.to(self.out_device) for d in data)
    
    def __len__(self):
        return 1
