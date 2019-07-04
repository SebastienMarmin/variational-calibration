import torch

class ProbabilityLaw(torch.nn.Module):

    def __init__(self, distribution):
        super(ProbabilityLaw, self).__init__()
        self.distrib = distribution
        #self.params = torch.nn.ParameterList()

    def make_parameter(self,string):
        data = getattr(self.distrib,string).data
        param = torch.nn.Parameter(data)
        #print("hoh")
        #print(data)
        
        #print(getattr(self.distrib,string))
        setattr(self.distrib,string,param)

        #self.params.append(param)